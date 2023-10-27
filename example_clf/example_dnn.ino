#include "dnn_model.h"

void setup() {
    Serial.begin(115200);

    while (!dnn_model.begin()) {
        Serial.print("Error in NN initialization: ");
        Serial.println(dnn_model.getErrorMessage());
    }
}

void loop() {
    
        float input[2] = {0.31699103, -2.38886223};
        float y_pred = dnn_model.predict(input);
        
        Serial.println("Expected output: ");
        Serial.println(-48.130339359863974);
        
        Serial.println("Predicted output:");
        Serial.println(y_pred);
        delay(1000);
    }
