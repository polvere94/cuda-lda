#!/bin/bash
echo "ciao";
for i in {50..1024}
do
	echo $i;
	./a $i;
done    
wait(10000);