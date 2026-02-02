#include <Braccio.h>
#include <Servo.h>

Servo base;
Servo shoulder;
Servo elbow;
Servo wrist_ver;
Servo wrist_rot;
Servo gripper;

// --- SAFE STARTUP POSITION ---
// Base: 90 (Center)
// Shoulder: 40 (Low/Resting)
// Elbow: 180 (Folded tight)
// Wrist_Ver: 170 (Tucked in)
// Wrist_Rot: 90 (Neutral)
// Gripper: 73 (Open)
// -----------------------------

void setup() {
  Braccio.begin();
  Serial.begin(9600);
  
  // Move to "Folded/Stable" position immediately on boot
  // Speed 20 is safe for startup
  Braccio.ServoMovement(30, 90, 45, 180, 180, 90, 73);
}

void loop() {
  if (Serial.available() > 0) {
    char startChar = Serial.read();
    
    if (startChar == '<') {
      // Parse the 6 integers from Python
      int m1 = Serial.parseInt(); // Base
      int m2 = Serial.parseInt(); // Shoulder
      int m3 = Serial.parseInt(); // Elbow
      int m4 = Serial.parseInt(); // Wrist Ver
      int m5 = Serial.parseInt(); // Wrist Rot
      int m6 = Serial.parseInt(); // Gripper
      
      // --- SAFETY LIMITS ---
      // Prevent the arm from smashing into the table or itself
      m2 = constrain(m2, 15, 165); 
      m3 = constrain(m3, 0, 180);
      m4 = constrain(m4, 0, 180);
      m6 = constrain(m6, 10, 73); // 10=Closed Hard, 73=Open
      
      // Move to the new target
      Braccio.ServoMovement(30, m1, m2, m3, m4, m5, m6);
    }
  }
}