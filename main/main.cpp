#include <cstdio>
#include <stdio.h>
#include <cstdlib>
#include "esp_log.h"
#include "model_data.h"

// wrapper de Espressif / TFLM
#include "tensorflow/lite/micro/micro_op_resolver.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"

#include "esp_heap_caps.h"
#include "esp_system.h"
#include "esp_err.h"

#include "esp_psram.h" 


static const char *TAG = "TFLM_SETUP";
constexpr int kTensorArenaSize = 700 * 1024;
static uint8_t *tensor_arena = nullptr;

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

    // Cargar modelo TFLite desde el arreglo generado
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

    // Resolver ops: usamos MicroMutableOpResolver
    static tflite::MicroMutableOpResolver<15> resolver;

    // Registrar las operaciones necesarias según el modelo
    resolver.AddQuantize();
    resolver.AddConv2D();
    resolver.AddRelu6();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    resolver.AddMean();
    resolver.AddFullyConnected();
    resolver.AddSoftmax();
    resolver.AddDequantize();


    // Crear intérprete
    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    ESP_LOGI(TAG, "Intérprete creado correctamente.");

    // Asignar tensores
    TfLiteStatus allocate_status = interpreter.AllocateTensors();
    if (allocate_status != kTfLiteOk) {
        ESP_LOGE(TAG, "Error al asignar tensores.");
        return;
    }

    ESP_LOGI(TAG, "Tensor arena asignada correctamente.");
    ESP_LOGI(TAG, "Setup completo. Modelo listo para inferencias.");
}
