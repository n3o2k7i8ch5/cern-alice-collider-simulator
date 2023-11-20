g++ root_event_part_loop.cpp `root-config --libs --cflags` -o run -lEG
./run pythia.root > raw_data
rm run
