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
#include "esp_camera.h"

#include "esp_wifi.h"
#include "esp_now.h"

#include "model_data.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"

#include "img_converters.h"


// Inclusiones para el PIR y SD
#include "driver/gpio.h"
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdspi_host.h"
#include "driver/spi_common.h"
#include <sys/stat.h>

static const char *TAG = "TESIS_CAM";


// Configuración de Pines
#define PIR_GPIO   GPIO_NUM_3
#define FLASH_GPIO GPIO_NUM_4

// Pines SPI para SD
#define PIN_NUM_MISO GPIO_NUM_2
#define PIN_NUM_MOSI GPIO_NUM_15
#define PIN_NUM_CLK  GPIO_NUM_14
#define PIN_NUM_CS   GPIO_NUM_13



static int photo_count = 0;  
static char current_session_dir[32];

// =========================================================
//                   CONFIGURACIÓN ESP-NOW
// =========================================================

// Dirección MAC del ESP32 Esclavo
uint8_t peer_mac_address[] = {0xEC, 0xE3, 0x34, 0xDA, 0xC5, 0xB0}; 

typedef struct {
    int class_id;
} class_data_t;

static bool esp_now_initialized = false;

// ===== CONFIGURACIÓN CAMARA & TFLM =====
#define CONFIG_XCLK_FREQ 8000000
#define CAM_WIDTH 320
#define CAM_HEIGHT 240
#define TARGET_SIZE 96
#define PART_BOUNDARY "123456789000000000000987654321"


static const char* _STREAM_CONTENT_TYPE = "multipart/x-mixed-replace;boundary=" PART_BOUNDARY;
static const char* _STREAM_BOUNDARY = "\r\n--" PART_BOUNDARY "\r\n";
static const char* _STREAM_PART = "Content-Type: image/jpeg\r\nContent-Length: %u\r\n\r\n";

constexpr int kTensorArenaSize = 700 * 1024;
static uint8_t *tensor_arena = nullptr;
static tflite::MicroInterpreter *interpreter = nullptr;
static TfLiteTensor *input = nullptr;

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
    gpio_set_level(FLASH_GPIO, 0);  

    // PIR
    gpio_reset_pin(PIR_GPIO);
    gpio_set_direction(PIR_GPIO, GPIO_MODE_INPUT);

    vTaskDelay(pdMS_TO_TICKS(500)); 
}


//Función de inicialización de SD
esp_err_t init_sdcard() {
    spi_bus_config_t bus_cfg = {
        .mosi_io_num = PIN_NUM_MOSI,
        .miso_io_num = PIN_NUM_MISO,
        .sclk_io_num = PIN_NUM_CLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
    };
    esp_err_t ret = spi_bus_initialize(SPI2_HOST, &bus_cfg, SPI_DMA_CH_AUTO);
    if (ret != ESP_OK) return ret;

    sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_config.gpio_cs = PIN_NUM_CS;
    slot_config.host_id = SPI2_HOST;

    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 5,
    };

    sdmmc_host_t host = SDSPI_HOST_DEFAULT();
    sdmmc_card_t *card;
    return esp_vfs_fat_sdspi_mount("/sdcard", &host, &slot_config, &mount_config, &card);
}

//Función para crear carpeta por sesión

void create_session_folder() {
    int session_id = 0;
    struct stat st;
    
    // Buscar el siguiente ID de sesión disponible
    do {
        session_id++;
        sprintf(current_session_dir, "/sdcard/s%d", session_id);
    } while (stat(current_session_dir, &st) == 0); 

    // Crear la carpeta
    if (mkdir(current_session_dir, 0775) == 0) {
        ESP_LOGI(TAG, "📁 Nueva sesión creada: %s", current_session_dir);
    } else {
        ESP_LOGE(TAG, "❌ Error al crear carpeta de sesión");
        strcpy(current_session_dir, "/sdcard");
    }
}



void save_fb_to_sd(camera_fb_t *fb, const char* label) {
    if (!fb) return;
    char path[128];
    
    sprintf(path, "%s/%d_%.3s.jpg", current_session_dir, photo_count++, label);

    FILE *file = fopen(path, "wb");
    if (!file) {
        ESP_LOGE(TAG, "Error al abrir archivo: %s", path);
        return;
    }
    
    fwrite(fb->buf, 1, fb->len, file);
    fclose(file);
    
    ESP_LOGI(TAG, "📸 Foto guardada: %s", path);
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

// Función de envío ESP-NOW 
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
static void run_inference_and_save(camera_fb_t *fb)
{
    if (!fmt2rgb888(fb->buf, fb->len, fb->format, rgb_buf)) return;

    // Redimensionar y Normalizar
    for (int y = 0; y < TARGET_SIZE; y++) {
        int iy = (int)(y * ((float)fb->height / TARGET_SIZE));
        for (int x = 0; x < TARGET_SIZE; x++) {
            int ix = (int)(x * ((float)fb->width / TARGET_SIZE));
            int src = (iy * fb->width + ix) * 3;
            int dst = (y * TARGET_SIZE + x) * 3;

            for (int c = 0; c < 3; c++) {
                // Obtenemos el valor 0-255
                float pixel_val = (float)rgb_buf[src + c];
                
                // Aplicamos la misma normalización del entrenamiento (/255.0)
                // Y ajustamos al input del modelo cuantizado
                float normalized_val = pixel_val / 255.0f;
                
                // Ajuste para el tensor INT8 (input->params contiene la escala del modelo)
                input->data.uint8[dst + c] = (uint8_t)(normalized_val / input->params.scale + input->params.zero_point);
            }
        }
    }
    // 4️⃣ Ejecutar inferencia
    if (interpreter->Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Error ejecutando inferencia");
        return;
    }

    // 5️⃣ Interpretar resultados
    TfLiteTensor *output = interpreter->output(0);
    int predicted_class = -1;
    float max_prob = -1.0f;

    for (int i = 0; i < 4 ; i++) {
        float prob = (output->data.uint8[i] - output->params.zero_point) * output->params.scale;
        if (prob > max_prob) {
            max_prob = prob;
            predicted_class = i;
        }
    }

    // VALIDACIÓN DE SEGURIDAD PARA LA SD
    if (predicted_class >= 0 && predicted_class <= 3) {
        std::string label = label_map[predicted_class];
        ESP_LOGI(TAG, "🧠 Objeto: %s (%.2f%%)", label.c_str(), max_prob * 100);
        
        send_class_by_espnow(predicted_class);
        gpio_set_level(FLASH_GPIO, 0);
        vTaskDelay(pdMS_TO_TICKS(100)); // Respiro para el voltaje
        
        // Guardar usando el label validado
        save_fb_to_sd(fb, label.c_str());
    } else {
        ESP_LOGE(TAG, "Clase predicha fuera de rango: %d", predicted_class);
    }
}

// ===== MAIN =====
extern "C" void app_main(void)
{
    // 0. Inicializar NVS
    esp_err_t ret = nvs_flash_init();
    if (ret == ESP_ERR_NVS_NO_FREE_PAGES || ret == ESP_ERR_NVS_NEW_VERSION_FOUND) {
        ESP_ERROR_CHECK(nvs_flash_erase());
        ret = nvs_flash_init();
    }
    
    ESP_ERROR_CHECK(esp_netif_init());
    ESP_ERROR_CHECK(esp_event_loop_create_default()); 
    wifi_init_config_t cfg = WIFI_INIT_CONFIG_DEFAULT();
    ESP_ERROR_CHECK(esp_wifi_init(&cfg));
    ESP_ERROR_CHECK(esp_wifi_set_mode(WIFI_MODE_STA));
    ESP_ERROR_CHECK(esp_wifi_start());
    
    init_espnow_master(); 
    init_pir_and_flash();

    if (init_sdcard() == ESP_OK) {
        create_session_folder();
        photo_count=0;
    } else {
        ESP_LOGE(TAG, "Fallo crítico: No se pudo montar la SD");
    }

    if (init_camera() != ESP_OK) return;
    alloc_buffers();
    init_tflite();

    ESP_LOGI(TAG, "✅ Sistema listo.");

    camera_fb_t *fb = NULL;
    while (1) {
        int pir = gpio_get_level(PIR_GPIO);

        if (pir == 1) {
            
            ESP_LOGI(TAG, "🚶 Movimiento detectado");

            //1. Encender luz
            gpio_set_level(FLASH_GPIO, 1); 

            //2. Esperar a que el sensor se adapte a la luz 
            vTaskDelay(pdMS_TO_TICKS(800));

            //3. Limpiar buffer

            camera_fb_t *fb_old = esp_camera_fb_get();
            if (fb_old) {
                esp_camera_fb_return(fb_old); 
                ESP_LOGD(TAG, "Buffer limpiado");
            }

            //4. Captura que se va a usar 
            camera_fb_t *fb = esp_camera_fb_get();
            if (fb) {
                run_inference_and_save(fb);
                esp_camera_fb_return(fb);
            } else {
                ESP_LOGE(TAG, "Fallo al capturar frame");
            }

            //5. Esperar para no saturar al sensor y los servos puedan trabajar tranquilos
            vTaskDelay(pdMS_TO_TICKS(4000));
        }

        vTaskDelay(pdMS_TO_TICKS(100));
    }

}