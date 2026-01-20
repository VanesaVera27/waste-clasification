#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "esp_psram.h"
#include "nvs_flash.h"
#include "camera_pins.h"

#include "freertos/FreeRTOS.h"
#include "freertos/task.h"
#include "esp_timer.h"
#include "esp_camera.h"
#include "esp_http_server.h"

#include "esp_wifi.h"
#include "esp_now.h"


#include "connect_wifi.h"

#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"

#include "img_converters.h"


//-------- inclusiones para el PIR
#include "driver/gpio.h"

#define PIR_GPIO   GPIO_NUM_14
#define FLASH_GPIO GPIO_NUM_4


static const char *TAG = "CAM+TFLM";

// =========================================================
//                   CONFIGURACIÓN ESP-NOW (NUEVO)
// =========================================================
// Dirección MAC del ESP32 Esclavo
// *****************************************************************
uint8_t peer_mac_address[] = {0xEC, 0xE3, 0x34, 0xDA, 0xC5, 0xB0}; 

// *****************************************************************

// Estructura de datos a enviar 
typedef struct {
    int class_id;
} class_data_t;

static bool esp_now_initialized = false;

// ===== CONFIGURACIÓN CAMARA =====
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

#define CONFIG_XCLK_FREQ 10000000
#define CAM_WIDTH 320
#define CAM_HEIGHT 240
#define TARGET_SIZE 96

// ===== CONFIGURACIÓN TENSORFLOW LITE MICRO =====
constexpr int kTensorArenaSize = 700 * 1024;
static uint8_t *tensor_arena = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input = nullptr;

// Buffers grandes en PSRAM
static uint8_t *rgb_buf = nullptr;
static uint8_t *resized_buf = nullptr;

std::map<int, std::string> label_map = {
    {0, "carton"},
    {1, "metal"},
    {2, "papel"},
    {3, "plastico"}
};

// ===== FUNCIONES =====

//Función de inicialización de PIR y FLASH
void init_pir_and_flash()
{
    // Flash
    gpio_reset_pin(FLASH_GPIO);
    gpio_set_direction(FLASH_GPIO, GPIO_MODE_OUTPUT);
    gpio_set_level(FLASH_GPIO, 0);  // apagado inicial

    // PIR
    gpio_reset_pin(PIR_GPIO);
    gpio_set_direction(PIR_GPIO, GPIO_MODE_INPUT);

    vTaskDelay(pdMS_TO_TICKS(500)); // estabilización
}

// Función de inicialización ESP-NOW 
void init_espnow_master() {
    // Inicializar ESP-NOW
    if (esp_now_init() != ESP_OK) {
        ESP_LOGE(TAG, "Error inicializando ESP-NOW");
        return;
    }
    esp_now_initialized = true;

    // Registrar el Peer (Esclavo)
    esp_now_peer_info_t peer_info;
    memset(&peer_info, 0, sizeof(peer_info));
    memcpy(peer_info.peer_addr, peer_mac_address, 6);
    peer_info.channel = 0; 
    peer_info.encrypt = false;

    if (esp_now_add_peer(&peer_info) != ESP_OK) {
        ESP_LOGE(TAG, "Error al agregar Peer");
        return;
    }
    ESP_LOGI(TAG, "ESP-NOW Maestro configurado y Peer Esclavo añadido.");
}

// Función de envío ESP-NOW (NUEVO)
void send_class_by_espnow(int class_index) {
    if (!esp_now_initialized) return;

    class_data_t data;
    data.class_id = class_index;

    esp_err_t result = esp_now_send(peer_mac_address, (uint8_t *) &data, sizeof(data));
    
    if (result == ESP_OK) {
        ESP_LOGI(TAG, "ESP-NOW enviado: Clase %d", class_index);
    } else {
        ESP_LOGE(TAG, "Error de envio ESP-NOW: %s", esp_err_to_name(result));
    }
}

// Inicializar cámara
static esp_err_t init_camera(void)
{
    camera_config_t config = {
        .pin_pwdn = CAM_PIN_PWDN,
        .pin_reset = CAM_PIN_RESET,
        .pin_xclk = CAM_PIN_XCLK,
        .pin_sccb_sda = CAM_PIN_SIOD,
        .pin_sccb_scl = CAM_PIN_SIOC,
        .pin_d7 = CAM_PIN_D7,
        .pin_d6 = CAM_PIN_D6,
        .pin_d5 = CAM_PIN_D5,
        .pin_d4 = CAM_PIN_D4,
        .pin_d3 = CAM_PIN_D3,
        .pin_d2 = CAM_PIN_D2,
        .pin_d1 = CAM_PIN_D1,
        .pin_d0 = CAM_PIN_D0,
        .pin_vsync = CAM_PIN_VSYNC,
        .pin_href = CAM_PIN_HREF,
        .pin_pclk = CAM_PIN_PCLK,

        .xclk_freq_hz = CONFIG_XCLK_FREQ,
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_JPEG,
        .frame_size = FRAMESIZE_QVGA,
        .jpeg_quality = 12,
        .fb_count = 1
    };

    esp_err_t err = esp_camera_init(&config);
    if (err != ESP_OK) {
        ESP_LOGE(TAG, "Error iniciando cámara: %s", esp_err_to_name(err));
    }
    return err;
}

// Inicializar buffers grandes en PSRAM
static void alloc_buffers() {
    if (esp_psram_is_initialized()) {
        rgb_buf = (uint8_t *)heap_caps_malloc(CAM_WIDTH * CAM_HEIGHT * 3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        resized_buf = (uint8_t *)heap_caps_malloc(TARGET_SIZE * TARGET_SIZE * 3, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        if (!rgb_buf || !resized_buf) {
            ESP_LOGE(TAG, "No se pudo asignar memoria en PSRAM");
            abort();
        }
        ESP_LOGI(TAG, "Buffers asignados en PSRAM");
    } else {
        rgb_buf = (uint8_t *)malloc(CAM_WIDTH * CAM_HEIGHT * 3);
        resized_buf = (uint8_t *)malloc(TARGET_SIZE * TARGET_SIZE * 3);
        if (!rgb_buf || !resized_buf) {
            ESP_LOGE(TAG, "No se pudo asignar memoria en RAM interna");
            abort();
        }
    }
}

// Inicializar modelo
static void init_tflite()
{
    ESP_LOGI(TAG, "Inicializando TensorFlow Lite Micro...");

    if (esp_psram_is_initialized()) {
        tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        ESP_LOGI(TAG, "Usando PSRAM para tensor arena.");
    } else {
        tensor_arena = (uint8_t *)malloc(kTensorArenaSize);
        ESP_LOGW(TAG, "PSRAM no detectada, usando RAM interna.");
    }

    const tflite::Model *model = tflite::GetModel(modelo_tflite);
    static tflite::MicroMutableOpResolver<15> resolver;
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddConv2D();
    resolver.AddRelu();
    resolver.AddRelu6();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    resolver.AddMean();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();

    static tflite::MicroInterpreter static_interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    interpreter = &static_interpreter;

    if (interpreter->AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "Error asignando tensores");
        return;
    }

    input = interpreter->input(0);
    ESP_LOGI(TAG, "Modelo inicializado correctamente.");
}

// Ejecutar inferencia
static void run_inference(camera_fb_t *fb)
{
    if (!interpreter || !fb) return;

    // 1️⃣ Convertir frame JPEG a RGB888
    if (!fmt2rgb888(fb->buf, fb->len, fb->format, rgb_buf)) {
        ESP_LOGE(TAG, "Error al convertir a RGB888");
        return;
    }

    // 2️⃣ Redimensionar a 96x96 con interpolación cercana y ajuste de contraste
    for (int y = 0; y < TARGET_SIZE; y++) {
        float src_y = y * ((float)fb->height / TARGET_SIZE);
        int iy = (int)src_y;
        for (int x = 0; x < TARGET_SIZE; x++) {
            float src_x = x * ((float)fb->width / TARGET_SIZE);
            int ix = (int)src_x;
            int src_index = (iy * fb->width + ix) * 3;
            int dst_index = (y * TARGET_SIZE + x) * 3;

            for (int c = 0; c < 3; c++) {
                int val = (int)(rgb_buf[src_index + c] * 1.1 + 10);  // contraste + brillo
                if (val > 255) val = 255;
                if (val < 0) val = 0;
                resized_buf[dst_index + c] = (uint8_t)val;
            }
        }
    }

    // 3️⃣ Copiar al tensor del modelo (uint8)
    memcpy(input->data.uint8, resized_buf, TARGET_SIZE * TARGET_SIZE * 3);

    // 4️⃣ Ejecutar inferencia
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Error ejecutando inferencia");
        return;
    }

    // 5️⃣ Interpretar resultados
    TfLiteTensor *output = interpreter->output(0);
    int predicted_class = -1;
    float max_prob = -1.0f;

    for (int i = 0; i < output->dims->data[1]; i++) {
        float prob = (float)output->data.uint8[i] / 255.0f; 
        if (prob > max_prob) {
            max_prob = prob;
            predicted_class = i;
        }
    }

    if (predicted_class >= 0) {
        ESP_LOGI(TAG, "🧠 Objeto detectado: %s (%.2f%%)",
                     label_map[predicted_class].c_str(), max_prob * 100);
        
        // 6️⃣ ENVIAR CLASE AL ESCLAVO POR ESP-NOW 
        send_class_by_espnow(predicted_class);
    }
}

// Handler HTTP con inferencia
esp_err_t jpg_stream_httpd_handler(httpd_req_t *req)
{
    camera_fb_t *fb = NULL;
    esp_err_t res = httpd_resp_set_type(req, _STREAM_CONTENT_TYPE);
    if (res != ESP_OK) return res;

    while (true) {
        fb = esp_camera_fb_get();
        if (!fb) {
            ESP_LOGE(TAG, "Error capturando frame");
            break;
        }

        run_inference(fb);

        res = httpd_resp_send_chunk(req, _STREAM_BOUNDARY, strlen(_STREAM_BOUNDARY));
        if (res == ESP_OK) {
            char part_buf[64];
            size_t hlen = snprintf(part_buf, 64, _STREAM_PART, fb->len);
            res = httpd_resp_send_chunk(req, part_buf, hlen);
        }
        if (res == ESP_OK)
            res = httpd_resp_send_chunk(req, (const char *)fb->buf, fb->len);

        esp_camera_fb_return(fb);
        if (res != ESP_OK) break;
    }

    return res;
}

httpd_uri_t uri_get = {
    .uri = "/",
    .method = HTTP_GET,
    .handler = jpg_stream_httpd_handler,
    .user_ctx = NULL
};

httpd_handle_t setup_server(void)
{
    httpd_config_t config = HTTPD_DEFAULT_CONFIG();
    httpd_handle_t server = NULL;

    if (httpd_start(&server, &config) == ESP_OK) {
        httpd_register_uri_handler(server, &uri_get);
        ESP_LOGI(TAG, "Servidor HTTP iniciado");
    }
    return server;
}

// ===== MAIN (FINAL) =====
extern "C" void app_main(void)
{
    // 0. Inicializar NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    
    // 1. Inicialización completa del stack de red (SOLO UNA VEZ)
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default()); 
    //esp_netif_create_default_wifi_sta(); 

    // 2. Inicialización base del Wi-Fi
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    
    // 3. Inicialización de ESP-NOW (solo la parte peer)
    init_espnow_master(); 

    //connect_wifi();

    // 5. Componentes de la aplicación
    if (init_camera() != ESP_OK) return;
    alloc_buffers();
    init_tflite();
    init_pir_and_flash();

    ESP_LOGI(TAG, "✅ Sistema listo: espnow en bucle");

    camera_fb_t *fb = NULL;
    while (1) {
        int pir = gpio_get_level(PIR_GPIO);

        if (pir == 1) {
            ESP_LOGI(TAG, "🚶 Movimiento detectado");

            gpio_set_level(FLASH_GPIO, 1);  // prender flash

            camera_fb_t *fb = esp_camera_fb_get();
            if (fb) {
                run_inference(fb);
                esp_camera_fb_return(fb);
            } else {
                ESP_LOGE(TAG, "Fallo al capturar frame");
            }

            gpio_set_level(FLASH_GPIO, 0);  // apagar flash

            vTaskDelay(pdMS_TO_TICKS(3000));
        }

        vTaskDelay(pdMS_TO_TICKS(100));
    }

}