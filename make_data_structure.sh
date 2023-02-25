#!/bin/bash

mkdir -p Data/NHTS_2017
mkdir -p Data/Generated_Data
mkdir -p Data/ACS_2021/Tract_Geometries

[ ! -f ACS2021_Table_Shells.csv ] ||
cp ACS2021_Table_Shells.csv Data/ACS_2021/ACS2021_Table_Shells.csv

if [ ! -f Data/ACS_2021/Tract_Geometries/tracts.zip ]; then
	curl -o Data/ACS_2021/Tract_Geometries/tracts.zip https://www2.census.gov/geo/tiger/GENZ2021/shp/cb_2021_us_tract_500k.zip
	echo "Tracts Downloaded"
else
	echo "Tracts Downloaded"
fi

if [ ! -f Data/ACS_2021/Tract_Geometries/cb_2021_us_bg_500k.shp ]; then
	unzip Data/ACS_2021/Tract_Geometries/tracts.zip -d Data/ACS_2021/Tract_Geometries
	echo "Tracts Unzipped"
else
	echo "Tracts Unzipped"
fi

if [ ! -f Data/NHTS_2017/csv.zip ]; then
	curl -o Data/NHTS_2017/csv.zip https://nhts.ornl.gov/assets/2016/download/csv.zip
	echo "NHTS Data Downloaded"
else
	echo "NHTS Data Downloaded"
fi

if [ ! -f Data/NHTS_2017/trippub.csv ]; then
	unzip Data/NHTS_2017/csv.zip -d Data/NHTS_2017
	echo "NHTS Data Unzipped"
else
	echo "NHTS Data Unzipped"
fi