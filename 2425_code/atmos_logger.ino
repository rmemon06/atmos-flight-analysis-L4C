#include <Wire.h>
#include <SPI.h>
#include <SD.h>
#include <Adafruit_MCP9808.h>      // Temperature Sensor
#include <Adafruit_MPL3115A2.h>    // Altimeter/Barometer

// --- Main I2C bus for Temp + Altimeter ---
#define MAIN_SDA A4
#define MAIN_SCL A5

// --- Secondary I2C bus for CO2 + Flow ---
TwoWire WireAux = TwoWire(1);
#define AUX_SDA D6
#define AUX_SCL D5

// --- SD card ---
#define SD_CS_PIN D10
#define LOG_FILE  "/atmos_log.csv"

// --- I2C Addresses ---
#define MCP9808_ADDR 0x18
#define COZIR_ADDR   0x41
#define FLOW_ADDR    0x40

// --- Sensor Objects ---
Adafruit_MCP9808 tempsensor = Adafruit_MCP9808();
Adafruit_MPL3115A2 altimeter = Adafruit_MPL3115A2();

// --- Globals ---
float baselineAltitude = 0;

// One-off I2C scan at boot (replaces the separate scanner sketch)
void scanBus(TwoWire &bus, const char *name) {
  Serial.print("I2C devices on ");
  Serial.print(name);
  Serial.print(": ");
  int nDevices = 0;
  for (byte address = 1; address < 127; address++) {
    bus.beginTransmission(address);
    if (bus.endTransmission() == 0) {
      Serial.print("0x");
      if (address < 16) Serial.print("0");
      Serial.print(address, HEX);
      Serial.print(" ");
      nDevices++;
    }
  }
  Serial.println(nDevices == 0 ? "none found" : "");
}

// Print a line to Serial and append it to the SD log
void logLine(const String &line) {
  Serial.println(line);
  File file = SD.open(LOG_FILE, FILE_APPEND);
  if (file) {
    file.println(line);
    file.close();   // close every line so data survives a power cut
  } else {
    Serial.println("Failed to open log file!");
  }
}

void setup() {
  Serial.begin(115200);
  delay(1000);

  Serial.println("Initializing sensors...");
  Serial.println("---------------------------------");

  // Start both I2C buses and list what's on them
  Wire.begin(MAIN_SDA, MAIN_SCL);
  if (!WireAux.begin(AUX_SDA, AUX_SCL)) {
    Serial.println("Aux I2C init failed!");
    while (1);
  }
  scanBus(Wire, "main bus");
  scanBus(WireAux, "aux bus");

  // Init temperature sensor
  if (!tempsensor.begin(MCP9808_ADDR)) {
    Serial.println("Couldn't find MCP9808 temp sensor!");
    while (1);
  }
  Serial.println("MCP9808 Temperature ... [OK]");
  tempsensor.shutdown_wake(0);

  // Init altimeter
  if (!altimeter.begin()) {
    Serial.println("Couldn't find MPL3115A2 altimeter!");
    while (1);
  }
  Serial.println("MPL3115A2 Altimeter ... [OK]");
  altimeter.setSeaPressure(1013.25); // hPa

  // Start flow sensor continuous measurement
  WireAux.beginTransmission(FLOW_ADDR);
  WireAux.write(0x10);
  WireAux.write(0x00);
  if (WireAux.endTransmission() != 0) {
    Serial.println("Failed to start flow measurement!");
  } else {
    Serial.println("Flow measurement started.");
  }
  delay(5);

  // Init SD card
  if (!SD.begin(SD_CS_PIN)) {
    Serial.println("SD card initialization failed!");
    while (1);
  }
  Serial.println("SD card ... [OK]");

  // Baseline altitude
  baselineAltitude = altimeter.getAltitude();
  Serial.print("Baseline altitude: ");
  Serial.println(baselineAltitude, 2);

  Serial.println("All sensors initialized. Starting data stream...");
  logLine("Timestamp(ms),Temp(C),Pressure(hPa),RelAlt(m),CO2(ppm),Flow(slm)");
}

void loop() {
  unsigned long timestamp = millis();

  // temp
  float temperature = tempsensor.readTempC();

  // altitude + pressure
  float currentAltitude = altimeter.getAltitude();
  float relativeAltitude = currentAltitude - baselineAltitude;
  float currentPressure = altimeter.getPressure(); // hPa

  // co2
  int co2_ppm = -1;
  WireAux.beginTransmission(COZIR_ADDR);
  WireAux.write(0x02);
  WireAux.endTransmission();
  WireAux.requestFrom(COZIR_ADDR, 2);
  if (WireAux.available() == 2) {
    uint8_t hi = WireAux.read();   // read bytes separately: C++ doesn't guarantee
    uint8_t lo = WireAux.read();   // the order of two read() calls in one expression
    co2_ppm = (hi << 8) | lo;
  } else {
    Serial.println("Warning: No CO2 data received!");
  }

  // flow rate
  float flowSLM = -999;
  WireAux.requestFrom(FLOW_ADDR, 3);
  if (WireAux.available() == 3) {
    uint8_t hi = WireAux.read();
    uint8_t lo = WireAux.read();
    WireAux.read(); // CRC, ignored
    uint16_t rawFlow = (hi << 8) | lo;
    flowSLM = 200.0f * ((float)rawFlow - 32768.0f) / 32768.0f; // TODO: check offset/scale against sensor datasheet
  } else {
    Serial.println("No flow data received!");
  }

  logLine(String(timestamp) + "," +
          String(temperature, 2) + "," +
          String(currentPressure, 2) + "," +
          String(relativeAltitude, 2) + "," +
          String(co2_ppm) + "," +
          String(flowSLM, 2));

  delay(1000);
}
