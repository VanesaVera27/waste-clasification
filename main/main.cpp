#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>

#include "esp_log.h"
#include "esp_system.h"
#include "esp_heap_caps.h"
#include "esp_psram.h"

#include "model_data.h"      
#include "imagen.h"         

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

extern "C" void app_main(void)
{
    ESP_LOGI(TAG, "=== Inicializando TensorFlow Lite Micro ===");

    // Asignar memoria para tensor arena
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
        ESP_LOGE(TAG, "Error: modelo no encontrado");
        return;
    }
    if (model->version() != TFLITE_SCHEMA_VERSION) {
        ESP_LOGE(TAG, "Versión de modelo incompatible: %d != %d", model->version(), TFLITE_SCHEMA_VERSION);
        return;
    }

    ESP_LOGI(TAG, "Modelo cargado correctamente (%u bytes).", modelo_tflite_len);

    // Resolver operaciones
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

    // Crear intérprete
    static tflite::MicroInterpreter interpreter(model, resolver, tensor_arena, kTensorArenaSize);
    if (interpreter.AllocateTensors() != kTfLiteOk) {
        ESP_LOGE(TAG, "Error al asignar tensores.");
        return;
    }

    ESP_LOGI(TAG, "Tensor arena asignada correctamente.");

    // Obtener tensor de entrada
    TfLiteTensor *input = interpreter.input(0);

    if (input->bytes != sizeof(test_jpg)) {
        ESP_LOGW(TAG, "Tamaño de imagen (%d) no coincide con el esperado (%d).", sizeof(test_jpg), input->bytes);
    }

    // Copiar imagen al tensor de entrada
    if (input->type == kTfLiteUInt8) {
        memcpy(input->data.uint8, test_jpg, sizeof(test_jpg));
    } else if (input->type == kTfLiteInt8) {
        for (int i = 0; i < sizeof(test_jpg); i++)
            input->data.int8[i] = (int)test_jpg[i] - 128;  // Normalización si el modelo es int8
    } else {
        ESP_LOGE(TAG, "Tipo de tensor no soportado.");
        return;
    }

    ESP_LOGI(TAG, "Ejecutando inferencia...");
    if (interpreter.Invoke() != kTfLiteOk) {
        ESP_LOGE(TAG, "Error al ejecutar inferencia.");
        return;
    }

    ESP_LOGI(TAG, "Inferencia completada correctamente.");

    // Procesar salida
    TfLiteTensor *output = interpreter.output(0);
    int predicted_class = -1;
    float max_prob = -1.0f;

    for (int i = 0; i < output->dims->data[1]; i++) {
        float prob = (float)output->data.uint8[i] / 255.0f;
        ESP_LOGI(TAG, "Clase %d (%s): %.4f", i, label_map[i].c_str(), prob);
        if (prob > max_prob) {
            max_prob = prob;
            predicted_class = i;
        }
    }

    if (predicted_class >= 0) {
        ESP_LOGI(TAG, "✅ Clase predicha: %s (%.2f%%)", 
                 label_map[predicted_class].c_str(), max_prob * 100);
    } else {
        ESP_LOGW(TAG, "No se pudo determinar clase.");
    }

    ESP_LOGI(TAG, "=== Fin de app_main ===");
}
