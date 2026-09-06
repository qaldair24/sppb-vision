/*
  SPPB-VISION | ESP32-S3 + MPU6050 lumbar L5

  Hardware:
    SDA -> GPIO 8
    SCL -> GPIO 9

  Funciones:
    - Punto de acceso Wi-Fi ESP32_SPPB (clave 12345678)
    - MPU6050 a 50 Hz
    - Transmisión UDP en vivo al puerto 4212
    - Calibración del giroscopio con control de quietud
    - Registro CSV de respaldo en LittleFS
    - Control HTTP desde SPPB-VISION

  Paquete de muestra:
    IMU,seq,ms_esp32,ms_prueba,grabando,fase,ax,ay,az,gx,gy,gz,tempC

  Paquete de evento:
    EVENT,etiqueta,ms_esp32,ms_prueba,fase
*/

#define ARDUINO_LOOP_STACK_SIZE 16384

#include <Adafruit_MPU6050.h>
#include <Adafruit_Sensor.h>
#include <FS.h>
#include <LittleFS.h>
#include <WebServer.h>
#include <WiFi.h>
#include <WiFiUdp.h>
#include <Wire.h>
#include <math.h>

// ---------------- Configuración ----------------
const char* AP_SSID = "ESP32_SPPB";
const char* AP_PASS = "12345678";

constexpr uint8_t SDA_PIN = 8;
constexpr uint8_t SCL_PIN = 9;
constexpr uint16_t SAMPLE_HZ = 50;
constexpr uint32_t SAMPLE_PERIOD_MS = 1000 / SAMPLE_HZ;
constexpr uint16_t LIVE_UDP_PORT = 4212;
constexpr size_t CSV_MAX_BYTES = 2 * 1024 * 1024;

const char* CSV_PATH = "/imu_l5.csv";
IPAddress BROADCAST_IP(192, 168, 4, 255);

// ---------------- Dispositivos ----------------
Adafruit_MPU6050 mpu;
WebServer server(80);
WiFiUDP liveUdp;

// ---------------- Estado ----------------
volatile bool mpuFound = false;
volatile bool loggingActive = false;
volatile bool calibrating = false;
volatile bool offsetsValid = false;
volatile bool storageFull = false;

uint8_t mpuAddress = 0x68;
uint32_t testStartMs = 0;
uint32_t sequenceNumber = 0;
uint32_t rowsWritten = 0;

float gyroBiasX = 0.0f;
float gyroBiasY = 0.0f;
float gyroBiasZ = 0.0f;

String patientId = "0000";
String testType = "chair";
String repetition = "1";
String placement = "L5";
String notes = "";
String testPhase = "idle";
String calibrationMessage = "Pendiente";

File csvFile;
SemaphoreHandle_t fsMutex = nullptr;
SemaphoreHandle_t udpMutex = nullptr;
TaskHandle_t samplerTaskHandle = nullptr;

// ---------------- Utilidades ----------------
String safeText(String value) {
  value.replace(",", "_");
  value.replace("\n", " ");
  value.replace("\r", " ");
  value.replace("\"", "'");
  value.replace("/", "-");
  value.replace("\\", "-");
  return value;
}

String jsonEscape(String value) {
  value.replace("\\", "\\\\");
  value.replace("\"", "\\\"");
  value.replace("\n", " ");
  value.replace("\r", " ");
  return value;
}

bool detectMpu() {
  if (mpu.begin(0x68, &Wire)) {
    mpuAddress = 0x68;
    return true;
  }
  if (mpu.begin(0x69, &Wire)) {
    mpuAddress = 0x69;
    return true;
  }
  return false;
}

void configureMpu() {
  mpu.setAccelerometerRange(MPU6050_RANGE_8_G);
  mpu.setGyroRange(MPU6050_RANGE_500_DEG);
  mpu.setFilterBandwidth(MPU6050_BAND_21_HZ);
}

uint32_t elapsedTestMs(uint32_t nowMs) {
  return loggingActive ? nowMs - testStartMs : 0;
}

void sendJson(bool ok, const String& message, int statusCode = 200) {
  String json = "{\"ok\":";
  json += ok ? "true" : "false";
  json += ",\"message\":\"";
  json += jsonEscape(message);
  json += "\"}";
  server.send(statusCode, "application/json; charset=utf-8", json);
}

String jsonStringValue(const String& body, const char* key) {
  String token = String("\"") + key + "\":";
  int start = body.indexOf(token);
  if (start < 0) return "";
  start += token.length();
  while (start < static_cast<int>(body.length()) && body[start] == ' ') start++;
  if (start >= static_cast<int>(body.length()) || body[start] != '"') return "";
  int finish = body.indexOf('"', start + 1);
  if (finish <= start) return "";
  return body.substring(start + 1, finish);
}

bool openCsvAppend() {
  if (csvFile) return true;
  csvFile = LittleFS.open(CSV_PATH, "a");
  return static_cast<bool>(csvFile);
}

void closeCsv() {
  if (csvFile) {
    csvFile.flush();
    csvFile.close();
  }
}

bool rebuildCsv() {
  if (xSemaphoreTake(fsMutex, pdMS_TO_TICKS(500)) != pdTRUE) return false;
  closeCsv();
  File file = LittleFS.open(CSV_PATH, "w");
  if (!file) {
    xSemaphoreGive(fsMutex);
    return false;
  }

  file.printf("# patient_id=%s\n", safeText(patientId).c_str());
  file.printf("# test=%s\n", safeText(testType).c_str());
  file.printf("# repetition=%s\n", safeText(repetition).c_str());
  file.printf("# placement=%s\n", safeText(placement).c_str());
  file.printf("# sample_hz=%u\n", SAMPLE_HZ);
  file.printf("# board=ESP32-S3-WROOM-1\n");
  file.printf("# imu=MPU6050\n");
  file.printf("# mpu_address=0x%02X\n", mpuAddress);
  file.printf("# offsets_valid=%s\n", offsetsValid ? "true" : "false");
  file.printf(
    "# gyro_bias_rad_s=%.7f,%.7f,%.7f\n",
    gyroBiasX,
    gyroBiasY,
    gyroBiasZ
  );
  if (notes.length()) file.printf("# notes=%s\n", safeText(notes).c_str());
  file.print(
    "node_id,seq,ms_esp32,ms_prueba,event,phase,"
    "ax_m_s2,ay_m_s2,az_m_s2,gx_rad_s,gy_rad_s,gz_rad_s,temp_c\n"
  );
  file.flush();
  file.close();
  rowsWritten = 0;
  storageFull = false;
  xSemaphoreGive(fsMutex);
  return true;
}

void appendCsvLine(const String& line) {
  if (storageFull) return;
  if (xSemaphoreTake(fsMutex, pdMS_TO_TICKS(50)) != pdTRUE) return;
  if (!openCsvAppend()) {
    xSemaphoreGive(fsMutex);
    return;
  }

  if (csvFile.size() + line.length() > CSV_MAX_BYTES) {
    storageFull = true;
    loggingActive = false;
    closeCsv();
    xSemaphoreGive(fsMutex);
    return;
  }

  csvFile.print(line);
  rowsWritten++;
  if (rowsWritten % 25 == 0) csvFile.flush();
  xSemaphoreGive(fsMutex);
}

void sendUdp(const String& message) {
  if (xSemaphoreTake(udpMutex, pdMS_TO_TICKS(20)) != pdTRUE) return;
  liveUdp.beginPacket(BROADCAST_IP, LIVE_UDP_PORT);
  liveUdp.write(
    reinterpret_cast<const uint8_t*>(message.c_str()),
    message.length()
  );
  liveUdp.endPacket();
  xSemaphoreGive(udpMutex);
}

void appendEvent(const String& label) {
  uint32_t nowMs = millis();
  uint32_t relativeMs = elapsedTestMs(nowMs);
  String cleanLabel = safeText(label);

  String packet = "EVENT,";
  packet += cleanLabel;
  packet += ",";
  packet += String(nowMs);
  packet += ",";
  packet += String(relativeMs);
  packet += ",";
  packet += safeText(testPhase);
  sendUdp(packet);

  if (loggingActive || label == "stop") {
    String line = "WAIST_LOCAL,";
    line += String(sequenceNumber);
    line += ",";
    line += String(nowMs);
    line += ",";
    line += String(relativeMs);
    line += ",";
    line += cleanLabel;
    line += ",";
    line += safeText(testPhase);
    line += ",,,,,,,\n";
    appendCsvLine(line);
  }
}

// ---------------- Calibración ----------------
bool calibrateGyro() {
  if (!mpuFound) {
    calibrationMessage = "MPU6050 no detectado";
    return false;
  }

  calibrating = true;
  offsetsValid = false;
  calibrationMessage = "Mantenga quieto el sensor lumbar";
  delay(SAMPLE_PERIOD_MS + 5);

  constexpr int SAMPLE_COUNT = 150;
  constexpr float MAX_GYRO_MAG = 0.30f;
  constexpr float MIN_ACCEL_MAG = 7.8f;
  constexpr float MAX_ACCEL_MAG = 11.8f;
  constexpr int MAX_MOVING_SAMPLES = 12;

  double sumGx = 0.0;
  double sumGy = 0.0;
  double sumGz = 0.0;
  int movingSamples = 0;

  for (int i = 0; i < SAMPLE_COUNT; i++) {
    sensors_event_t acceleration, gyro, temperature;
    mpu.getEvent(&acceleration, &gyro, &temperature);

    float gyroMag = sqrtf(
      gyro.gyro.x * gyro.gyro.x +
      gyro.gyro.y * gyro.gyro.y +
      gyro.gyro.z * gyro.gyro.z
    );
    float accelMag = sqrtf(
      acceleration.acceleration.x * acceleration.acceleration.x +
      acceleration.acceleration.y * acceleration.acceleration.y +
      acceleration.acceleration.z * acceleration.acceleration.z
    );
    if (
      gyroMag > MAX_GYRO_MAG ||
      accelMag < MIN_ACCEL_MAG ||
      accelMag > MAX_ACCEL_MAG
    ) {
      movingSamples++;
    }

    sumGx += gyro.gyro.x;
    sumGy += gyro.gyro.y;
    sumGz += gyro.gyro.z;
    delay(20);
  }

  if (movingSamples > MAX_MOVING_SAMPLES) {
    calibrating = false;
    calibrationMessage = "Movimiento detectado durante la calibración";
    return false;
  }

  gyroBiasX = static_cast<float>(sumGx / SAMPLE_COUNT);
  gyroBiasY = static_cast<float>(sumGy / SAMPLE_COUNT);
  gyroBiasZ = static_cast<float>(sumGz / SAMPLE_COUNT);
  offsetsValid = true;
  calibrating = false;
  calibrationMessage = "Calibración OK";
  rebuildCsv();
  return true;
}

// ---------------- Adquisición a 50 Hz ----------------
void samplerTask(void* parameter) {
  TickType_t lastWake = xTaskGetTickCount();
  while (true) {
    vTaskDelayUntil(&lastWake, pdMS_TO_TICKS(SAMPLE_PERIOD_MS));
    if (!mpuFound || calibrating) continue;

    sensors_event_t acceleration, gyro, temperature;
    mpu.getEvent(&acceleration, &gyro, &temperature);
    uint32_t nowMs = millis();
    uint32_t relativeMs = elapsedTestMs(nowMs);
    uint32_t seq = sequenceNumber++;

    float gx = gyro.gyro.x - (offsetsValid ? gyroBiasX : 0.0f);
    float gy = gyro.gyro.y - (offsetsValid ? gyroBiasY : 0.0f);
    float gz = gyro.gyro.z - (offsetsValid ? gyroBiasZ : 0.0f);

    String packet;
    packet.reserve(230);
    packet += "IMU,";
    packet += String(seq);
    packet += ",";
    packet += String(nowMs);
    packet += ",";
    packet += String(relativeMs);
    packet += ",";
    packet += (loggingActive ? "1" : "0");
    packet += ",";
    packet += safeText(testPhase);
    packet += ",";
    packet += String(acceleration.acceleration.x, 6);
    packet += ",";
    packet += String(acceleration.acceleration.y, 6);
    packet += ",";
    packet += String(acceleration.acceleration.z, 6);
    packet += ",";
    packet += String(gx, 6);
    packet += ",";
    packet += String(gy, 6);
    packet += ",";
    packet += String(gz, 6);
    packet += ",";
    packet += String(temperature.temperature, 2);
    sendUdp(packet);

    if (loggingActive) {
      String line = "WAIST_LOCAL,";
      line += String(seq);
      line += ",";
      line += String(nowMs);
      line += ",";
      line += String(relativeMs);
      line += ",,";
      line += safeText(testPhase);
      line += ",";
      line += String(acceleration.acceleration.x, 6);
      line += ",";
      line += String(acceleration.acceleration.y, 6);
      line += ",";
      line += String(acceleration.acceleration.z, 6);
      line += ",";
      line += String(gx, 6);
      line += ",";
      line += String(gy, 6);
      line += ",";
      line += String(gz, 6);
      line += ",";
      line += String(temperature.temperature, 2);
      line += "\n";
      appendCsvLine(line);
    }
  }
}

// ---------------- API HTTP ----------------
void registerRoutes() {
  server.on("/", HTTP_GET, []() {
    String page =
      "<!doctype html><html><head><meta charset='utf-8'>"
      "<meta name='viewport' content='width=device-width,initial-scale=1'>"
      "<title>SPPB IMU L5</title></head><body style='font-family:Arial;margin:24px'>"
      "<h2>SPPB-VISION | IMU lumbar L5</h2>"
      "<p>ESP32-S3-WROOM-1 + MPU6050</p>"
      "<p>Estado del MPU6050: <b>";
    page += mpuFound ? "OK" : "NO DETECTADO";
    page +=
      "</b></p><p>Frecuencia: 50 Hz</p>"
      "<p>Datos en vivo: UDP 4212</p>"
      "<p>El control automático se realiza desde SPPB-VISION.</p>"
      "<p><a href='/status'>Ver estado JSON</a> | "
      "<a href='/download'>Descargar CSV</a></p></body></html>";
    server.send(200, "text/html; charset=utf-8", page);
  });

  server.on("/status", HTTP_GET, []() {
    String json = "{";
    json += "\"mpu_found\":";
    json += mpuFound ? "true" : "false";
    json += ",\"logging\":";
    json += loggingActive ? "true" : "false";
    json += ",\"calibrating\":";
    json += calibrating ? "true" : "false";
    json += ",\"offsets_valid\":";
    json += offsetsValid ? "true" : "false";
    json += ",\"storage_full\":";
    json += storageFull ? "true" : "false";
    json += ",\"sample_hz\":";
    json += String(SAMPLE_HZ);
    json += ",\"stream_port\":";
    json += String(LIVE_UDP_PORT);
    json += ",\"rows\":";
    json += String(rowsWritten);
    json += ",\"phase\":\"";
    json += jsonEscape(testPhase);
    json += "\",\"calibration_message\":\"";
    json += jsonEscape(calibrationMessage);
    json += "\"}";
    server.send(200, "application/json; charset=utf-8", json);
  });

  server.on("/meta", HTTP_POST, []() {
    if (loggingActive) {
      sendJson(false, "Detenga la grabación antes de cambiar metadatos.", 409);
      return;
    }
    if (!server.hasArg("plain")) {
      sendJson(false, "Falta el cuerpo JSON.", 400);
      return;
    }
    String body = server.arg("plain");
    String value;
    value = jsonStringValue(body, "patient_id");
    if (value.length()) patientId = value;
    value = jsonStringValue(body, "test_type");
    if (value.length()) testType = value;
    value = jsonStringValue(body, "repetition");
    if (value.length()) repetition = value;
    value = jsonStringValue(body, "placement");
    if (value.length()) placement = value;
    notes = jsonStringValue(body, "notes");
    testPhase = "idle";
    offsetsValid = false;
    calibrationMessage = "Pendiente";
    rebuildCsv();
    sendJson(true, "Metadatos guardados.");
  });

  server.on("/prepare", HTTP_POST, []() {
    if (loggingActive) {
      sendJson(false, "El IMU ya está grabando.", 409);
      return;
    }
    bool ok = calibrateGyro();
    sendJson(ok, calibrationMessage, ok ? 200 : 422);
  });

  server.on("/start", HTTP_POST, []() {
    if (loggingActive) {
      sendJson(true, "El IMU ya está grabando.");
      return;
    }
    if (!mpuFound) {
      sendJson(false, "MPU6050 no detectado.", 503);
      return;
    }

    bool usePrepared = server.hasArg("prepared") && server.arg("prepared") == "1";
    if ((!usePrepared || !offsetsValid) && !calibrateGyro()) {
      sendJson(false, calibrationMessage, 422);
      return;
    }

    closeCsv();
    if (!openCsvAppend()) {
      sendJson(false, "No se pudo abrir el CSV en LittleFS.", 500);
      return;
    }
    testPhase = "rest";
    testStartMs = millis();
    loggingActive = true;
    appendEvent("start");
    sendJson(true, "Grabación IMU iniciada.");
  });

  server.on("/mark", HTTP_POST, []() {
    if (!loggingActive) {
      sendJson(false, "El IMU no está grabando.", 409);
      return;
    }
    String label = server.hasArg("label") ? server.arg("label") : "marker";
    if (label.startsWith("camera_MO")) testPhase = "chair";
    if (label == "camera_R5") testPhase = "post";
    appendEvent(label);
    sendJson(true, "Evento guardado: " + safeText(label));
  });

  server.on("/stop", HTTP_POST, []() {
    if (loggingActive) {
      testPhase = "post";
      appendEvent("stop");
      loggingActive = false;
    }
    if (xSemaphoreTake(fsMutex, pdMS_TO_TICKS(500)) == pdTRUE) {
      closeCsv();
      xSemaphoreGive(fsMutex);
    }
    sendJson(true, "Grabación IMU detenida.");
  });

  server.on("/clear", HTTP_POST, []() {
    loggingActive = false;
    testPhase = "idle";
    offsetsValid = false;
    storageFull = false;
    calibrationMessage = "Pendiente";
    gyroBiasX = gyroBiasY = gyroBiasZ = 0.0f;
    rebuildCsv();
    sendJson(true, "Registro IMU reiniciado.");
  });

  server.on("/download", HTTP_GET, []() {
    if (xSemaphoreTake(fsMutex, pdMS_TO_TICKS(500)) != pdTRUE) {
      server.send(503, "text/plain", "Archivo ocupado.");
      return;
    }
    closeCsv();
    File file = LittleFS.open(CSV_PATH, "r");
    if (!file) {
      xSemaphoreGive(fsMutex);
      server.send(500, "text/plain", "No se pudo abrir el CSV.");
      return;
    }
    String filename = safeText(patientId) + "_IMU_L5_STS.csv";
    server.sendHeader(
      "Content-Disposition",
      "attachment; filename=\"" + filename + "\""
    );
    server.streamFile(file, "text/csv");
    file.close();
    xSemaphoreGive(fsMutex);
  });
}

void setup() {
  Serial.begin(115200);
  delay(400);
  Serial.println("\nSPPB-VISION | IMU L5");

  fsMutex = xSemaphoreCreateMutex();
  udpMutex = xSemaphoreCreateMutex();
  if (!LittleFS.begin(true)) {
    Serial.println("ERROR: LittleFS no pudo iniciar.");
  } else {
    Serial.println("LittleFS OK");
  }

  Wire.begin(SDA_PIN, SCL_PIN);
  Wire.setClock(400000);
  mpuFound = detectMpu();
  if (mpuFound) {
    configureMpu();
    Serial.print("MPU local OK en 0x");
    Serial.println(mpuAddress, HEX);
  } else {
    Serial.println("MPU local NO ENCONTRADA");
  }

  rebuildCsv();

  WiFi.mode(WIFI_AP);
  WiFi.softAP(AP_SSID, AP_PASS);
  liveUdp.begin(0);
  Serial.print("Red Wi-Fi: ");
  Serial.println(AP_SSID);
  Serial.print("IP: ");
  Serial.println(WiFi.softAPIP());
  Serial.print("UDP en vivo: ");
  Serial.println(LIVE_UDP_PORT);

  registerRoutes();
  server.begin();
  Serial.println("HTTP activo");

  xTaskCreatePinnedToCore(
    samplerTask,
    "imuSampler",
    8192,
    nullptr,
    1,
    &samplerTaskHandle,
    1
  );
}

void loop() {
  server.handleClient();
  delay(1);
}
