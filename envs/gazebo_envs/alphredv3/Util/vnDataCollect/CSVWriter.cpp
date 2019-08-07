//
// Created by jeff on 12/12/18.
//
#include "CSVWriter.h"

using namespace vn::math;

void CSVWriter::addRowData(double sysTime, uint64_t startupTime, vec3f rates, vec3f accels) {
    imuDataFile << fixed << sysTime << delimeter << startupTime << delimeter;
    imuDataFile << accels.x << delimeter << accels.y << delimeter << accels.z << delimeter;
    imuDataFile << rates.x << delimeter << rates.y << delimeter << rates.z;
    imuDataFile << "\n";
    linesCount++;
}

void CSVWriter::openFile() {
    // Open the file in truncate mode if first line else in Append Mode
    imuDataFile.open(fileName, std::ios::out | (linesCount ? std::ios::app : std::ios::trunc));
}

void CSVWriter::closeFile() {
    imuDataFile.close();
}