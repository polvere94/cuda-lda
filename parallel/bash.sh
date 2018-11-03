#!/bin/bash
echo "ciao";
for i in {10..50}
do
	echo $i;
	./a $i;
done    
wait(10000);