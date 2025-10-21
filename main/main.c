#include <stdio.h>
#include "model_data.h"
#include "esp_log.h"

static const char *TAG = "MODEL_TEST";

void app_main(void)
{
    ESP_LOGI(TAG, "Iniciando carga del modelo...");
    ESP_LOGI(TAG, "Tamaño del modelo: %u bytes", modelo_tflite_len);

    // Solo para verificar primeros bytes del modelo
    for (int i = 0; i < 8; i++) {
        printf("0x%02x ", modelo_tflite[i]);
    }
    printf("\nModelo cargado correctamente en memoria.\n");
}
