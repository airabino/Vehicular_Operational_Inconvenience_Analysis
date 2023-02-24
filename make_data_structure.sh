#!/bin/bash

mkdir -p Data/NHTS_2017
mkdir -p Data/ACS_2021

[ ! -f ACS2021_Table_Shells.csv ] || cp ACS2021_Table_Shells.csv Data/ACS_2021/ACS2021_Table_Shells.csv
