#!/bin/bash

# Настройки
N=10000
T=4
CHUNK_SIZES=(10 50 100)
SCHEDULE_TYPES=("static" "dynamic" "guided" "auto" "runtime")
OUTPUT_FILE="measurements_schedule"
BIN_DIR="bin"
SRC_FILE="src/test_sc.cpp"

# Создаем директорию для бинарников
mkdir -p $BIN_DIR

# Очищаем файл результатов
> $OUTPUT_FILE

# Функция для измерения времени
measure() {
	local prog=$1
	local min_time=999999
	for i in {1..5}; do
    	time=$(OMP_NUM_THREADS=$T ./$prog $N | grep -oP 'Time: \K\d+\.\d+')
    	if (( $(echo "$time < $min_time" | bc -l) )); then
        	min_time=$time
    	fi
	done
	echo $min_time
}

# Основной цикл измерений
declare -A results

for type in "${SCHEDULE_TYPES[@]}"; do
	echo "Тестирование $type..."
	if [[ "$type" == @("static"|"dynamic"|"guided") ]]; then
    	for chunk in "${CHUNK_SIZES[@]}"; do
        	# Компиляция
        	bin_name="$BIN_DIR/test_${type}_${chunk}"
        	g++ -Wall -Wextra -fopenmp -O3 $SRC_FILE -o $bin_name \
            	-DSCHELDUE_TYPE=$type \
            	-DCHUNK_SIZE=$chunk

        	# Измерение
        	results[$type]+=" $(measure $bin_name)"
    	done
	else

    	# Компиляция для auto и runtime
    	bin_name="$BIN_DIR/test_${type}"
    	g++ -Wall -Wextra -fopenmp -O3 $SRC_FILE -o $bin_name \
        	-DSCHELDUE_TYPE=$type

    	# Измерение
    	results[$type]=$(measure $bin_name)
	fi
done

# Запись результатов в файл
echo "N=$N T=$T" | tee -a $OUTPUT_FILE
for type in "${SCHEDULE_TYPES[@]}"; do
	echo "${type}: ${results[$type]}" | tee -a $OUTPUT_FILE
done

echo "Результаты сохранены в $OUTPUT_FILE"
