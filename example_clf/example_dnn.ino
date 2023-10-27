#include "dnn_model.h"

void setup() {
    Serial.begin(115200);

    while (!dnn_model.begin()) {
        Serial.print("Error in NN initialization: ");
        Serial.println(sineNN.getErrorMessage());
    }
}

void loop() {
    
        float input[2] = {0.3552, -0.2074};
        float y_pred = dnn_model.predict(input);
        Serial.println("Expected output: ");
        Serial.print(12.8171)
        Serial.println("Predicted output:");
        Serial.print(y_pred)
        delay(1000);
    }
}