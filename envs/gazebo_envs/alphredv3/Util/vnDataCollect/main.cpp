#include <iostream>
//#include <fstream>
#include <chrono>
#include <vn/sensors.h> // This header file is to get access to VectorNav sensors
#include <vn/thread.h>  // This header file includes the sleep function
#include "CSVWriter.h"

using namespace std;
using namespace vn::math;
using namespace vn::sensors;
using namespace vn::protocol::uart;
using namespace vn::xplat;

void initializeIMU(VnSensor &vn100s, const string sPort, const uint32_t sBaud);
void startBinaryAsyncDataStream(VnSensor &vn100s, void* userData);
void binaryAsyncMessageCallback(void *userData, Packet& p, size_t index);



int main(int argc, char *argv[])
{
    const string SensorPort = "/dev/ttyUSB0";   // Linux format for virtual (USB) serial port
    //const uint32_t SensorBaudrate = 115200;     // 115200 Default baud rate for vectornav sensors
    const uint32_t SensorBaudrate = 921600;     // 921600 fastest baud rate for vectornav sensors

    VnSensor vs;    // Create VnSensor object and use it to connect to the sensor
    initializeIMU(vs, SensorPort, SensorBaudrate);  // Connect to IMU
    CSVWriter imuDataWriter("vnIMUData.csv");
    imuDataWriter.openFile();
    startBinaryAsyncDataStream(vs, &imuDataWriter); // Start streaming data!
    try
    {
        cout << "Sleeping while collecting data..." << endl;
        Thread::sleepSec(60);
    }
    catch(...)
    {
        cout << "Caught an exception!";
    }
    imuDataWriter.closeFile();
    vs.unregisterAsyncPacketReceivedHandler();

    vs.disconnect();
    return 0;
}


void initializeIMU(VnSensor &vn100s, const string sPort, const uint32_t sBaud)
{
    // Acceptable Baud rates : [9600, 19200, 38400, 57600, 115200, 128000, 230400, 460800, 921600]
    // Keep baud rate high, otherwise the binary async output may not be able to output at the highest rate
    vn100s.connect(sPort, sBaud);

    //This changes the baudrate of the serial communication to the sensor
    // Seems like the imu has a tendancy to reset the baud rate so watch out for that.
    const uint32_t FasterBaudrate = 921600;
    vn100s.changeBaudRate(FasterBaudrate);
    cout << "Baudrate changed to: " << FasterBaudrate << endl;

    string mn = vn100s.readModelNumber();   // Print the model of the sensor
    cout << "Model Number: " << mn << endl;
}

void startBinaryAsyncDataStream(VnSensor &vn100s, void* userData)
{
    // Set up the sensor with Binary Output Configuration, a way to constantly stream info to the PC.
    // Note: Common group is combo of commonly used outputs from other groups, and it is better to just use that if possible to reduce packet size.
    // Note: When using the Binary OR (|) flags, the data will output in the order of the groups, not the order you type the flags
    BinaryOutputRegister bor(
            ASYNCMODE_PORT1,    // Communicate through serial port 1
            1,                  // Rate divisor; data rate = 800 / (rate divisor)
            COMMONGROUP_TIMESTARTUP | COMMONGROUP_ANGULARRATE | COMMONGROUP_ACCEL,   // 'Common group': 4.4 in VN-100s manual
            TIMEGROUP_NONE,     // 'Time group':    4.5 in VN-100s manual
            IMUGROUP_NONE,      // 'IMU group':     4.6 in VN-100s manual
            GPSGROUP_NONE,      // No GPS on VN-100s, only on 200/300
            ATTITUDEGROUP_NONE, // 'Attitude group': 4.7 in VN-100s manual
            INSGROUP_NONE);     // No INS on VN-100s, only on 200/300
    // Writes to Binary Output 1 register, starting the async data
    vn100s.writeBinaryOutput1(bor);
    // Registers a callback method for notification when a new asynchronous data packet is received.
    vn100s.registerAsyncPacketReceivedHandler(userData, binaryAsyncMessageCallback);
}

//void binaryAsyncMessageCallback(void *userData, Packet& p, size_t index)
//{
//    // Check to see if the packet type is Binary (as opposed to ASCII)
//    if(p.type() == Packet::TYPE_BINARY)
//    {
//        // Check that the packet is compatible, ie. all the configured outputs are there
//        if(!p.isCompatible(
//                COMMONGROUP_TIMESTARTUP | COMMONGROUP_ANGULARRATE | COMMONGROUP_ACCEL,
//                TIMEGROUP_NONE,
//                IMUGROUP_NONE,
//                GPSGROUP_NONE,
//                ATTITUDEGROUP_NONE,
//                INSGROUP_NONE)) {
//            cout << "Dang, binary packet is not compatible!" << endl;
//            return;
//        }
//        double sysTime {std::chrono::duration_cast<std::chrono::duration<double>>(std::chrono::high_resolution_clock::now().time_since_epoch()).count()};
//        uint64_t timeStartup = p.extractUint64();
//        vec3f rates = p.extractVec3f();
//        vec3f accel = p.extractVec3f();
////        cout << fixed << "Time Startup: " << timeStartup << "; Sys Time: " << sysTime << "; Comp. Accel: " << accel << ";\tComp. Ang. Rates: " << rates << endl;
//        CSVWriter *writerPtr = static_cast<CSVWriter*>(userData);
//        writerPtr->addRowData(sysTime, timeStartup, rates, accel);
//    }
//    else
//        cout << "Yikes, not reading binary packets!" << endl;
//}
void binaryAsyncMessageCallback(void *userData, Packet& p, size_t index) {
    if(p.type() == Packet::TYPE_BINARY)
    {
    double sysTime{std::chrono::duration_cast<std::chrono::duration<double>>(
            std::chrono::high_resolution_clock::now().time_since_epoch()).count()};
    uint64_t timeStartup = p.extractUint64();
    vec3f rates = p.extractVec3f();
    vec3f accel = p.extractVec3f();
//        cout << fixed << "Time Startup: " << timeStartup << "; Sys Time: " << sysTime << "; Comp. Accel: " << accel << ";\tComp. Ang. Rates: " << rates << endl;
    CSVWriter *writerPtr = static_cast<CSVWriter *>(userData);
    writerPtr->addRowData(sysTime, timeStartup, rates, accel);
    }
    //else
    //    cout << "Yikes, didn't get a binary packet!" <<endl;
}
