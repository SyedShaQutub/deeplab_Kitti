#!/bin/bash
# Copyright 2018 The TensorFlow Authors All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
#
# Script to download and preprocess the PASCAL VOC 2012 dataset.
#
# Usage:
#   bash ./download_and_convert_voc2012.sh
#
# The folder structure is assumed to be:
#  + datasets
#     - build_data.py
#     - build_voc2012_data.py
#     - download_and_convert_voc2012.sh
#     - remove_gt_colormap.py
#     + pascal_voc_seg
#       + VOCdevkit
#         + VOC2012
#           + JPEGImages
#           + SegmentationClass
#

# Exit immediately if a command exits with a non-zero status.
set -e

CURRENT_DIR=$(pwd)
WORK_DIR="./kitti_seg"
dest="data_semantics"
mkdir -p "${WORK_DIR}"
cd "${WORK_DIR}"
mkdir -p "${dest}"

# Download the images.
BASE_URL="https://s3.eu-central-1.amazonaws.com/avg-kitti/data_semantics.zip"
FILENAME="data_semantics.zip"

if [ ! -f "${FILENAME}" ]; then
  echo "Downloading ${FILENAME} to ${WORK_DIR}"
  wget -nd -c "${BASE_URL}"
fi

echo "Uncompressing ${FILENAME}"
unzip -qq -o "${FILENAME}" -d "${dest}" 

cd "${CURRENT_DIR}"

# Root path for Kitti dataset.
KITTI_ROOT="${WORK_DIR}/data_semantics/training"
mkdir -p "${KITTI_ROOT}/Segmentation_list"

# Remove the colormap in the ground truth annotations.
SEG_FOLDER="${KITTI_ROOT}/semantic"
SEMANTIC_SEG_FOLDER="${KITTI_ROOT}/SegmentationClassRaw"

echo "preparing the data set into list format to create tfrecords"

python ./kitti_data_split.py \
  --IMAGE_FOLDER="${KITTI_ROOT}/image_2" \
  --LIST_FOLDER="${KITTI_ROOT}/Segmentation_list"

echo "Removing the color map in ground truth annotations..."
python ./remove_gt_colormap_kitti.py \
  --original_gt_folder="${SEG_FOLDER}" \
  --output_dir="${SEMANTIC_SEG_FOLDER}"

# Build TFRecords of the dataset.
# First, create output directory for storing TFRecords.
OUTPUT_DIR="${WORK_DIR}/tfrecord"
mkdir -p "${OUTPUT_DIR}"

IMAGE_FOLDER="${KITTI_ROOT}/image_2"
LIST_FOLDER="${KITTI_ROOT}/Segmentation_list"

echo "Converting Kitti dataset..."
python ./build_kitti_data.py \
  --image_folder="${IMAGE_FOLDER}" \
  --semantic_segmentation_folder="${SEMANTIC_SEG_FOLDER}" \
  --list_folder="${LIST_FOLDER}" \
  --image_format="png" \
  --output_dir="${OUTPUT_DIR}"
