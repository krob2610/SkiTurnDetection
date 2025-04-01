DEVICE_MAP = {
    "4ea9b8e2f62a4cf7b5813f0c5f364589": "Sony_Xperia_XA",
    "e2480295bb6d4d5bb73cc2bebd958f31": "HUAWEI_Y7",
    "6792f50a55cf43f988e107130f644285": "Sony_Xperia_M5",
    "9917e7609a9a4e609c230e53860f7599": "HONOR_8X",
    "e759d4f0f1544fd6a8d76f68cf467a13": "OPPO_AX7",
    "b7c9cce36c0f435a97966b44246f5955": "LENOVO_K6_NOTE",
    "03f99d7af5d44d959af480e95a4b720b": "LG_K10",
    "63ad695bbf604f6f8dc819258bf1a096": "HUAWEI_MATE",
    "78b99b8517ac4838aeec69198d9a795f": "HUAWEI_P9_LITE",
}

DEVICE_NUMBER = {
    "HUAWEI_Y7": "1",
    "Sony_Xperia_XA": "2",
    "HUAWEI_MATE": "3",
    "LG_K10": "4",
    "HUAWEI_P9_LITE": "5",
    "Sony_Xperia_M5": "6",
    "OPPO_AX7": "7",
    "LENOVO_K6_NOTE": "8",
    "HONOR_8X": "9",
}

INPUT_PATH = "input_data"
NEW_INPUT = "!new_input"
DATA_PATH = "data"
SENSORS = [
    "Metadata",
    "Accelerometer",
    "Gravity",
    "Orientation",
    "Magnetometer",
    "Location",
    "Annotation",
    "TotalAcceleration",
    "Rules",
    "MagnetometerUncalibrated",
    "AccelerometerUncalibrated",
]

SNESORS_CSV = [
    "Accelerometer",
    "AccelerometerUncalibrated",
    "Gravity",
    "Gyroscope",
    "GyroscopeUncalibrated",
    "Location",
    "Magnetometer",
    "MagnetometerUncalibrated",
    "Orientation",
    "TotalAcceleration",
]

# sensors that will be included in final DataFrame
FINAL_DF_SENSORS = [
    "Accelerometer",
    "Gravity",
    "Gyroscope",
    "GyroscopeUncalibrated",
    "Magnetometer",
    "MagnetometerUncalibrated",
    "Orientation",  
    "TotalAcceleration",
]

TIME = "time"
