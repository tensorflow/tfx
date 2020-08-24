from google.cloud import pubsub_v1

import datetime
import logging
from google.cloud import pubsub_v1
import constants
import os

import apache_beam as beam
from apache_beam.options.pipeline_options import PipelineOptions
import apache_beam.transforms.window as window

class WriteBatches(beam.DoFn):
    def __init__(self, output_path):
        self.output_path = output_path

    def process(self, batch, window=beam.DoFn.WindowParam):
        """Write one batch per file to a Google Cloud Storage bucket. """
        beam_window_start = window.start.to_utc_datetime()
        beam_window_end = window.end.to_utc_datetime()
        
        date_window_end = beam_window_end.strftime("%Y-%m-%d")
       
        ts_format = "%H:%M"
        time_window_start = beam_window_start.strftime(ts_format)
        time_window_end = beam_window_end.strftime(ts_format)
        
        filename = "-".join([time_window_start, time_window_end, 'data'])
        datedir = os.path.join(self.output_path, date_window_end)
        filepath =  os.path.join(datedir, filename + '.csv')
        
        if not os.path.exists(datedir):
            os.mkdir(datedir)
        
        with open(filepath, mode="w+") as f:
            # Tempfix for debugging purposes.
            f.write('pickup_community_area,fare,trip_start_month,trip_start_hour,'
                    'trip_start_day,trip_start_timestamp,pickup_latitude,pickup_longitude,'
                    'dropoff_latitude,dropoff_longitude,trip_miles,pickup_census_tract,'
                    'dropoff_census_tract,payment_type,company,trip_seconds,'
                    'dropoff_community_area,tips,big_tipper\n')
            
            for element in batch:
                f.write("{}\n".format(element.data.rstrip()))


def run(input_topic, output_path, window_size=60, allowed_lateness=60):
    pipeline_options = PipelineOptions(
        ['--streaming'], streaming=True)
    
    if not os.path.exists(output_path):
        os.mkdir(output_path)
    
    with beam.Pipeline(options=pipeline_options) as pipeline:
        (
            pipeline
            | "Read PubSub Messages" >> beam.io.ReadFromPubSub(
                topic=input_topic,
                with_attributes=True,
                timestamp_attribute='event_time')
   
            | "Window into Fixed Intervals" >> beam.WindowInto(
                window.FixedWindows(window_size), 
                allowed_lateness=allowed_lateness)
            
      
            # Use a dummy key to group the elements in the same window.
            | "Add Dummy Key" >> beam.Map(lambda elem: (None, elem))
            | "Groupby" >> beam.GroupByKey()
            | "Abandon Dummy Key" >> beam.MapTuple(lambda _, val: val)
            
            | "Write to File" >> beam.ParDo(WriteBatches(output_path))
        )

def main():
    logging.getLogger().setLevel(logging.INFO)

    publisher = pubsub_v1.PublisherClient()
    input_topic_path = publisher.topic_path(constants.PROJECT_ID, 
                                            constants.INPUT_TOPIC_NAME)
    
    run(
        input_topic_path,
        constants.OUTPUT_PATH,
        window_size=constants.WINDOW_DURATION,
        allowed_lateness=constants.WINDOW_ALLOWED_LATENESS
    )

if __name__ == "__main__":  # noqa
    main()
