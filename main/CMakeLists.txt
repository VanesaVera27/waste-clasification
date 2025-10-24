idf_component_register(
    SRCS 
        "main.cpp"
        "model_data.cc"
	    "connect_wifi.c"
    INCLUDE_DIRS 
        "."
    PRIV_REQUIRES 
        spi_flash 
        esp_psram 
        freertos
    REQUIRES 
        esp_wifi esp-tflite-micro nvs_flash esp_timer esp_http_server 
)