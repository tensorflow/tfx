# Copyright 2019 Google LLC. All Rights Reserved.
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
"""Generate avro file from Chicago taxi csv data."""

import fastavro
import pandas as pd


def get_schema():
  """Returns schema for Chicago taxi dataset."""
  name_to_types = {
      'company': 'string',
      'dropoff_census_tract': 'float',
      'dropoff_community_area': 'float',
      'dropoff_latitude': 'float',
      'dropoff_longitude': 'float',
      'fare': 'float',
      'payment_type': 'string',
      'pickup_census_tract': 'float',
      'pickup_community_area': 'int',
      'pickup_latitude': 'float',
      'pickup_longitude': 'float',
      'tips': 'float',
      'trip_miles': 'float',
      'trip_seconds': 'float',
      'trip_start_day': 'int',
      'trip_start_hour': 'int',
      'trip_start_month': 'int',
      'trip_start_timestamp': 'int'
  }

  # Allow every column to accept null types
  fields = [{'name': name, 'type': [col_type, 'null']}
            for name, col_type in name_to_types.items()]

  return {'name': 'Chicago Taxi dataset', 'type': 'record', 'fields': fields}


def generate_avro(src_file: str, output_file: str):
  """Generates avro file based on src file.

  Args:
    src_file: path to Chicago taxi dataset.
    output_file: output path for avro file.
  """
  df = pd.read_csv(src_file)
  # Replaces NaN's with None's for avroWriter to interpret null values
  df = df.where((pd.notnull(df)), None)

  records = df.to_dict(orient='records')

  parsed_schema = fastavro.parse_schema(get_schema())
  with open(output_file, 'wb') as f:
    fastavro.writer(f, parsed_schema, records)


if __name__ == '__main__':
  generate_avro('../csv/data.csv', 'data2.avro')
