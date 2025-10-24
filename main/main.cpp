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

#include "connect_wifi.h"

#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"

static const char *TAG = "CAM+TFLM";

// ===== CONFIGURACIÓN CAMARA =====
#define PART_BOUNDARY "123456789000000000000987654321"
static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

#define CONFIG_XCLK_FREQ 20000000

// ===== CONFIGURACIÓN TENSORFLOW LITE MICRO =====
constexpr int kTensorArenaSize = 700 * 1024;
static uint8_t *tensor_arena = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input = nullptr;

std::map<int, std::string> label_map = {
    {0, "carton"},
    {1, "metal"},
    {2, "papel"},
    {3, "plastico"}
};

// ===== FUNCIONES =====

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

// Ejecutar inferencia sobre una imagen
static void run_inference(camera_fb_t *fb)
{
    if (!interpreter || !fb) return;

    // ⚠️ IMPORTANTE: convertir imagen JPEG a RGB y redimensionar si hace falta
    // Tu modelo debe esperar imágenes del mismo tamaño que usaste para entrenar (ej: 96x96)
    // Aquí asumimos que ya lo adaptaste para eso.
    // Si el modelo es uint8 (cuantizado):
    uint8_t *input_data = input->data.uint8;

    // Simulamos copiar los primeros bytes del frame como ejemplo
    // (en tu caso deberías decodificar JPEG -> RGB y escalar)
    size_t bytes_to_copy = std::min((size_t)input->bytes, fb->len);
    memcpy(input_data, fb->buf, bytes_to_copy);

    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Error ejecutando inferencia");
        return;
    }

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

    if (predicted_class >= 0)
        ESP_LOGI(TAG, "🧠 Objeto detectado: %s (%.2f%%)", label_map[predicted_class].c_str(), max_prob * 100);
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

        // Ejecutar inferencia sobre el frame actual
        run_inference(fb);

        // Enviar frame al navegador
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
    .user_ctx = NULL};

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

// ===== MAIN =====
extern "C" void app_main(void)
{
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }

    connect_wifi();
    if (!wifi_connect_status) {
        ESP_LOGE(TAG, "No se pudo conectar al WiFi.");
        return;
    }

    if (init_camera() != ESP_OK) return;
    init_tflite();
    setup_server();

    ESP_LOGI(TAG, "✅ Sistema listo: cámara + modelo funcionando");
}
