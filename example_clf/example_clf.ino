#include "random_forest_clf.h"

Eloquent::ML::Port::RandomForest clf;

void setup() {
    Serial.begin(115200);
}

void loop() {
    float irisSample[4] = {6.2, 2.8, 4.8, 1.8};

    Serial.print("Predicted class (you should see '2'): ");
    Serial.println(clf.predict(irisSample));
    delay(1000);
}