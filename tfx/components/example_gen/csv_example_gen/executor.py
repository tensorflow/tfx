# Lint as: python2, python3
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
"""Generic TFX CSV example gen executor."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
from typing import Any, Dict, Iterable, List, Text

import absl
import apache_beam as beam
import tensorflow as tf
from tfx_bsl.coders import csv_decoder

from tfx import types
from tfx.components.example_gen.base_example_gen_executor import BaseExampleGenExecutor
from tfx.components.example_gen.base_example_gen_executor import INPUT_KEY
from tfx.types import artifact_utils
from tfx.utils import io_utils


@beam.typehints.with_input_types(List[csv_decoder.CSVCell],
                                 List[csv_decoder.ColumnInfo])
@beam.typehints.with_output_types(tf.train.Example)
class _ParsedCsvToTfExample(beam.DoFn):
  """A beam.DoFn to convert a parsed CSV line to a tf.Example."""

  def __init__(self):
    self._column_handlers = None

  def _process_column_infos(self, column_infos: List[csv_decoder.ColumnInfo]):
    column_handlers = []
    for column_info in column_infos:
      # pylint: disable=g-long-lambda
      if column_info.type == csv_decoder.ColumnType.INT:
        handler_fn = lambda csv_cell: tf.train.Feature(
            int64_list=tf.train.Int64List(value=[int(csv_cell)]))
      elif column_info.type == csv_decoder.ColumnType.FLOAT:
        handler_fn = lambda csv_cell: tf.train.Feature(
            float_list=tf.train.FloatList(value=[float(csv_cell)]))
      elif column_info.type == csv_decoder.ColumnType.STRING:
        handler_fn = lambda csv_cell: tf.train.Feature(
            bytes_list=tf.train.BytesList(value=[csv_cell]))
      else:
        handler_fn = None
      column_handlers.append((column_info.name, handler_fn))

    self._column_handlers = column_handlers

  def process(
      self, csv_cells: List[csv_decoder.CSVCell],
      column_infos: List[csv_decoder.ColumnInfo]) -> Iterable[tf.train.Example]:
    if not self._column_handlers:
      self._process_column_infos(column_infos)

    # skip blank lines.
    if not csv_cells:
      return

    if len(csv_cells) != len(self._column_handlers):
      raise ValueError('Invalid CSV line: {}'.format(csv_cells))

    feature = {}
    for csv_cell, (column_name, handler_fn) in zip(csv_cells,
                                                   self._column_handlers):
      if not csv_cell:
        feature[column_name] = tf.train.Feature()
        continue
      if not handler_fn:
        raise ValueError(
            'Internal error: failed to infer type of column {} while it'
            'had at least some values {}'.format(column_name, csv_cell))
      feature[column_name] = handler_fn(csv_cell)
    yield tf.train.Example(features=tf.train.Features(feature=feature))


@beam.ptransform_fn
@beam.typehints.with_input_types(beam.Pipeline)
@beam.typehints.with_output_types(tf.train.Example)
def _CsvToExample(  # pylint: disable=invalid-name
    pipeline: beam.Pipeline,
    input_dict: Dict[Text, List[types.Artifact]],
    exec_properties: Dict[Text, Any],  # pylint: disable=unused-argument
    split_pattern: Text) -> beam.pvalue.PCollection:
  """Read CSV files and transform to TF examples.

  Note that each input split will be transformed by this function separately.

  Args:
    pipeline: beam pipeline.
    input_dict: Input dict from input key to a list of Artifacts.
      - input_base: input dir that contains csv data. csv files must have header
        line.
    exec_properties: A dict of execution properties.
    split_pattern: Split.pattern in Input config, glob relative file pattern
      that maps to input files with root directory given by input_base.

  Returns:
    PCollection of TF examples.

  Raises:
    RuntimeError: if split is empty or csv headers are not equal.
  """
  input_base_uri = artifact_utils.get_single_uri(input_dict[INPUT_KEY])
  csv_pattern = os.path.join(input_base_uri, split_pattern)
  absl.logging.info(
      'Processing input csv data {} to TFExample.'.format(csv_pattern))

  csv_files = tf.io.gfile.glob(csv_pattern)
  if not csv_files:
    raise RuntimeError(
        'Split pattern {} does not match any files.'.format(csv_pattern))

  column_names = io_utils.load_csv_column_names(csv_files[0])
  for csv_files in csv_files[1:]:
    if io_utils.load_csv_column_names(csv_files) != column_names:
      raise RuntimeError(
          'Files in same split {} have different header.'.format(csv_pattern))

  parsed_csv_lines = (
      pipeline
      | 'ReadFromText' >> beam.io.ReadFromText(
          file_pattern=csv_pattern, skip_header_lines=1)
      | 'ParseCSVLine' >> beam.ParDo(csv_decoder.ParseCSVLine(delimiter=',')))
  # TODO(b/155997704) clean this up once tfx_bsl makes a release.
  if getattr(csv_decoder, 'PARSE_CSV_LINE_YIELDS_RAW_RECORDS', False):
    # parsed_csv_lines is the following tuple (parsed_lines, raw_records)
    # we only want the parsed_lines.
    parsed_csv_lines |= 'ExtractParsedCSVLines' >> beam.Keys()
  column_infos = beam.pvalue.AsSingleton(
      parsed_csv_lines
      | 'InferColumnTypes' >> beam.CombineGlobally(
          csv_decoder.ColumnTypeInferrer(column_names, skip_blank_lines=True)))

  return (parsed_csv_lines
          | 'ToTFExample' >> beam.ParDo(_ParsedCsvToTfExample(), column_infos))


class Executor(BaseExampleGenExecutor):
  """Generic TFX CSV example gen executor."""

  def GetInputSourceToExamplePTransform(self) -> beam.PTransform:
    """Returns PTransform for CSV to TF examples."""
    return _CsvToExample
