#!/bin/bash

mkdir -p Data/NHTS_2017
mkdir -p Data/ACS_2021/Tract_Geometries

[ ! -f ACS2021_Table_Shells.csv ] || cp ACS2021_Table_Shells.csv Data/ACS_2021/ACS2021_Table_Shells.csv

curl -o Data/ACS_2021/Tract_Geometries/tracts.zip https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_bg_500k.zip

unzip Data/ACS_2021/Tract_Geometries/tracts.zip -d Data/ACS_2021/Tract_Geometries

curl -o Data/NHTS_2017/csv.zip https://nhts.ornl.gov/assets/2016/download/csv.zip

unzip Data/NHTS_2017/csv.zip -d Data/NHTS_2017