#include "esp_camera.h"
#include "esp_log.h"
#include "nvs_flash.h"
#include "driver/gpio.h"

// SD por SPI (IMPORTANTE)
#include "esp_vfs_fat.h"
#include "sdmmc_cmd.h"
#include "driver/sdspi_host.h"
#include "driver/spi_common.h"

#include "camera_pins.h"

static const char *TAG = "AUTO_CAM";

// Flash (cambiar si hay conflicto)
#define FLASH_GPIO GPIO_NUM_4

// Pines SPI (ajustables)
#define PIN_NUM_MISO GPIO_NUM_2
#define PIN_NUM_MOSI GPIO_NUM_15
#define PIN_NUM_CLK  GPIO_NUM_14
#define PIN_NUM_CS   GPIO_NUM_13

static int photo_count = 0;

// ============================
// FLASH
// ============================
void init_flash()
{
    gpio_reset_pin(FLASH_GPIO);
    gpio_set_direction(FLASH_GPIO, GPIO_MODE_OUTPUT);
    gpio_set_level(FLASH_GPIO, 0);
}

// ============================
// SD por SPI (ESTABLE)
// ============================
esp_err_t init_sdcard()
{
    esp_err_t ret;

    spi_bus_config_t bus_cfg = {
        .mosi_io_num = PIN_NUM_MOSI,
        .miso_io_num = PIN_NUM_MISO,
        .sclk_io_num = PIN_NUM_CLK,
        .quadwp_io_num = -1,
        .quadhd_io_num = -1,
    };

    ret = spi_bus_initialize(SPI2_HOST, &bus_cfg, SPI_DMA_CH_AUTO);
    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "Error init SPI");
        return ret;
    }

    sdspi_device_config_t slot_config = SDSPI_DEVICE_CONFIG_DEFAULT();
    slot_config.gpio_cs = PIN_NUM_CS;
    slot_config.host_id = SPI2_HOST;

    esp_vfs_fat_sdmmc_mount_config_t mount_config = {
        .format_if_mount_failed = false,
        .max_files = 5,
    };

    sdmmc_host_t host = SDSPI_HOST_DEFAULT();

    sdmmc_card_t *card;
    ret = esp_vfs_fat_sdspi_mount("/sdcard", &host, &slot_config, &mount_config, &card);

    if (ret != ESP_OK) {
        ESP_LOGE(TAG, "❌ Error montando SD");
        return ret;
    }

    ESP_LOGI(TAG, "✅ SD OK");
    return ESP_OK;
}

// ============================
// CÁMARA (TU CONFIG)
// ============================
esp_err_t init_camera()
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

        .xclk_freq_hz = 10000000,
        .ledc_timer = LEDC_TIMER_0,
        .ledc_channel = LEDC_CHANNEL_0,
        .pixel_format = PIXFORMAT_JPEG,
        .frame_size = FRAMESIZE_QXGA,
        .jpeg_quality = 10,
        .fb_count = 1
    };

    return esp_camera_init(&config);
}

// ============================
// GUARDAR FOTO
// ============================
void save_photo()
{
    camera_fb_t *fb = esp_camera_fb_get();
    if (!fb) {
        ESP_LOGE(TAG, "Error capturando");
        return;
    }

    char path[64];
    sprintf(path, "/sdcard/photo_%d.jpg", photo_count++);

    FILE *file = fopen(path, "wb");
    if (!file) {
        ESP_LOGE(TAG, "Error archivo");
        esp_camera_fb_return(fb);
        return;
    }

    fwrite(fb->buf, 1, fb->len, file);
    fclose(file);

    ESP_LOGI(TAG, "📸 %s", path);

    esp_camera_fb_return(fb);
}

// ============================
// MAIN
// ============================
extern "C" void app_main(void)
{
    ESP_ERROR_CHECK(nvs_flash_init());

    init_flash();

    if (init_camera() != ESP_OK) {
        ESP_LOGE(TAG, "Error cámara");
        return;
    }

    if (init_sdcard() != ESP_OK) {
        ESP_LOGE(TAG, "Error SD");
        return;
    }



    ESP_LOGI(TAG, "🚀 Listo - foto cada 3 segundos");

    while (1) {
        gpio_set_level(FLASH_GPIO, 1); 
        vTaskDelay(pdMS_TO_TICKS(900)); 
        
        camera_fb_t *fb_discard = esp_camera_fb_get();
        esp_camera_fb_return(fb_discard);

        save_photo(); 
        
        gpio_set_level(FLASH_GPIO, 0);
        vTaskDelay(pdMS_TO_TICKS(3000));
    }
}