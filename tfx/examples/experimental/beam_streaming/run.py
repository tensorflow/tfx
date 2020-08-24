from threading import Thread
import time
import materialize_stream
import simulate_taxi_data_stream

def main():
    Thread(target=simulate_taxi_data_stream.main).start()
    Thread(target=materialize_stream.main).start()

if __name__ == '__main__':
    main()