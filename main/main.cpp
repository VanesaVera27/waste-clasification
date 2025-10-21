#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "esp_psram.h"

#include "model_data.h" 

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/c/common.h"

static const char *TAG = "TFLM_MAIN";
constexpr int kTensorArenaSize = 700 * 1024;
static uint8_t *tensor_arena = nullptr;

// Mapeo de clases
std::map<int, std::string> label_map = {
    {0, "carton"},
    {1, "metal"},
    {2, "papel"},
    {3, "plastico"}
};

// Imagen de prueba (rellena con valores dummy, 96x96x3)
uint8_t test_image[96 * 96 * 3];

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "Inicializando TensorFlow Lite Micro...");

    if (esp_psram_is_initialized()) {
        tensor_arena = (uint8_t *)heap_caps_malloc(kTensorArenaSize, MALLOC_CAP_SPIRAM | MALLOC_CAP_8BIT);
        ESP_LOGI(TAG, "PSRAM detectada y usada para tensor arena.");
    } else {
        tensor_arena = (uint8_t *)malloc(kTensorArenaSize);
        ESP_LOGW(TAG, "PSRAM no detectada, usando RAM interna.");
    }

    if (!tensor_arena) {
        ESP_LOGE(TAG, "Error al asignar memoria para tensor arena");
        return;
    }

    // Cargar modelo
    const tflite::Model *model = tflite::GetModel(modelo_tflite);
    if (!model) {
        ESP_LOGE(TAG, "No se pudo obtener el modelo (model == nullptr)");
        return;
    }
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Versión de modelo incompatible: %d != %d", model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    ESP_LOGI(TAG, "Modelo cargado correctamente. Tamaño: %u bytes", modelo_tflite_len);

    // Resolver ops
    static tflite::MicroMutableOpResolver<15> resolver;
    resolver.AddQuantize();
    resolver.AddDequantize();
    resolver.AddConv2D();
    resolver.AddRelu6();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    resolver.AddMean();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();

    // Crear intérprete
    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    ESP_LOGI(TAG, "Intérprete creado correctamente.");

    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "Error al asignar tensores.");
        return;
    }
    ESP_LOGI(TAG, "Tensor arena asignada correctamente. Setup completo.");

    // Copiar imagen de prueba al tensor de entrada
    TfLiteTensor* input = interpreter.input(0);
    if (input->type == kTfLiteUInt8) {
        for (int i = 0; i < 96*96*3; i++) {
            input->data.uint8[i] = test_image[i]; 
        }
    }

    ESP_LOGI(TAG, "Iniciando inferencia de prueba...");
    if (interpreter.Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Error al ejecutar inferencia.");
        return;
    }
    ESP_LOGI(TAG, "Inferencia ejecutada correctamente.");

    // Procesar salida
    TfLiteTensor* output = interpreter.output(0);
    int predicted_class = 0;
    float max_prob = 0.0f;

    for (int i = 0; i < output->dims->data[1]; i++) {
        float prob = static_cast<float>(output->data.uint8[i]) / 255.0f;
        ESP_LOGI(TAG, "Clase %d (%s) -> %f", i, label_map[i].c_str(), prob);
        if (prob > max_prob) {
            max_prob = prob;
            predicted_class = i;
        }
    }

    ESP_LOGI(TAG, "Clase predicha: %s con probabilidad: %f",
             label_map[predicted_class].c_str(), max_prob);

    ESP_LOGI(TAG, "app_main finalizado.");
}
