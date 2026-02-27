#include <Braccio.h>
#include <Servo.h>

Servo base;
Servo shoulder;
Servo elbow;
Servo wrist_ver;
Servo wrist_rot;
Servo gripper;

// Buffer for incoming commands
const byte numChars = 32;
char receivedChars[numChars];
boolean newData = false;

void setup() {
  // 1. Initialize Braccio in Safety Position
  // M1=90, M2=45, M3=180, M4=180, M5=90, M6=10
  Braccio.begin();
  Braccio.ServoMovement(30, 180, 45, 0, 180, 90, 73);
  
  // 2. Start High-Speed Serial
  Serial.begin(115200);
  Serial.println("READY");
}

void loop() {
  recvWithStartEndMarkers();
  if (newData == true) {
    processCommand();
    newData = false;
  }
}

// Non-blocking Serial Reader
void recvWithStartEndMarkers() {
  static boolean recvInProgress = false;
  static byte ndx = 0;
  char startMarker = 'P';
  char endMarker = '\n';
  char rc;

  while (Serial.available() > 0 && newData == false) {
    rc = Serial.read();

    if (recvInProgress == true) {
      if (rc != endMarker) {
        receivedChars[ndx] = rc;
        ndx++;
        if (ndx >= numChars) {
          ndx = numChars - 1;
        }
      } else {
        receivedChars[ndx] = '\0'; // Terminate string
        recvInProgress = false;
        ndx = 0;
        newData = true;
      }
    } else if (rc == startMarker) {
      recvInProgress = true;
    }
  }
}

void processCommand() {
  // Expected format: "90,45,180,180,90,10"
  int m1, m2, m3, m4, m5, m6;
  
  // Parse Integers
  char *strtokIndx; 
  
  strtokIndx = strtok(receivedChars, ",");      
  m1 = atoi(strtokIndx);
  
  strtokIndx = strtok(NULL, ","); 
  m2 = atoi(strtokIndx);
  
  strtokIndx = strtok(NULL, ","); 
  m3 = atoi(strtokIndx);
  
  strtokIndx = strtok(NULL, ","); 
  m4 = atoi(strtokIndx);
  
  strtokIndx = strtok(NULL, ","); 
  m5 = atoi(strtokIndx);
  
  strtokIndx = strtok(NULL, ","); 
  m6 = atoi(strtokIndx);

  // EXECUTE MOVE
  // Step Delay 10 = Maximum Safe Speed
  Braccio.ServoMovement(30, m1, m2, m3, m4, m5, m6);
  
  // Acknowledge (Optional, might slow down loop if verbose)
  // Serial.println("OK"); 
}