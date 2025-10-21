idf_component_register(SRCS "model_data.cc" "main.cpp"
                       PRIV_REQUIRES spi_flash esp_psram freertos
                       INCLUDE_DIRS "."
                       REQUIRES esp-tflite-micro)
