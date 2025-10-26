import numpy as np
import matplotlib.pyplot as plt

# Параметры ПСП
code_length = 511
chip_rate = 511e3
T_prn = 1e-3

SNR = 10        # в дБ

# Генерация ПСП сигнала l1of

def generate_l1of():
    # Регистр: [1, 2, 3, 4, 5, 6, 7, 8, 9]
    # 1 - вход (x⁹), 9 - выход (младший бит) ПСП снимается с 7-й ячейки
    # Начальное состояние: все единицы
    register = [1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    prn_code = np.zeros(code_length)
    
    for i in range(code_length):
        # ПСП снимается с 7-й ячейки (индекс 6 в Python)
        prn_code[i] = register[6]  # 7-я ячейка
        
        # Обратная связь: x⁵ + x⁹
        feedback = register[8] ^ register[4]  # 5-я ячейка XOR 1-я ячейка
        
        # СДВИГ ВПРАВО
        register = [feedback] + register[0:8]
    
    return prn_code

# Генерация шума

def generate_awgn(snr_db, signal_power=1.0):
    # Перевод SNR из дБ в линейный масштаб
    snr_linear = 10 ** (snr_db / 10)
    
    # Расчет мощности шума
    noise_power = signal_power / snr_linear
    
    # Генерация гауссовского шума
    noise = np.random.normal(0, np.sqrt(noise_power), code_length)
    
    return noise

def add_noise_to_signal(signal, snr_db):
    # Расчет мощности сигнала
    signal_power = np.mean(signal ** 2)
    
    # Генерация шума
    noise = generate_awgn(snr_db, signal_power)
    
    # Добавление шума к сигналу
    noisy_signal = signal + noise
    
    return noisy_signal, noise

# Для удобной записи массива в строку

def output_of_elements(signal, num_chips):
    result = ""
    for i in range(num_chips):
        chip_value = signal[i]
        chip_int = int(chip_value)
        chip_str = str(chip_int)
        result = result + chip_str
    return result

print("Генерация ПСП L1OF ГЛОНАСС с шумом SNR = 10 дБ")
print("=" * 50)

# Генерируем ПСП
l1of_code = generate_l1of()

# Проверяем начальную последовательность
first_9_chips = output_of_elements(l1of_code, 9)
print(f"Первые 9 чипов ПСП: {first_9_chips}")
print(f"Ожидается: 111111100")

# Добавляем шум с SNR = 10 дБ
noisy_signal, noise = add_noise_to_signal(l1of_code, SNR)

print(f"\nРезультаты для SNR = 10 дБ:")
print(f"Мощность сигнала: {np.mean(l1of_code**2):.3f}") # np.mean среднее арифметическое
print(f"Мощность шума: {np.mean(noise**2):.6f}")

print(f"\nСтатистики шума:")
print(f"Мат. ожидание: {np.mean(noise):.6f}")
print(f"Дисперсия: {np.var(noise):.6f}") # np.var - функция, которая вычисляет дисперсию
print(f"Среднеквадратичное отклонение: {np.std(noise):.6f}") # np.std = sqrt(np.var) т.е. СКО

# Статистика ПСП
ones = np.sum(l1of_code)
zeros = len(l1of_code) - ones
print(f"\nСтатистика ПСП:")
print(f"Единиц: {int(ones)} ({ones/code_length*100:.1f}%)")
print(f"Нулей: {int(zeros)} ({zeros/code_length*100:.1f}%)")

# Создаем график
plt.figure(figsize=(12, 6))

# Верхний график - сравнение сигналов
plt.subplot(2, 1, 1)
plt.plot(l1of_code[:30], 'bo-', linewidth=2, markersize=4, label='Чистый сигнал')
plt.plot(noisy_signal[:30], 'ro-', linewidth=1, markersize=2, alpha=0.7, label='Сигнал + шум')
plt.title('Сравнение чистого сигнала и сигнала с шумом')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(True)

plt.subplot(2, 1, 2)
noise_component = noisy_signal[:30] - l1of_code[:30]
plt.plot(noise_component, 'g-', linewidth=1, label='Шум')  # Зеленая линия
plt.title('Шум')
plt.xlabel('Отсчеты')
plt.ylabel('Амплитуда')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()





# Процедура поиска сигнала

delta_F = (2/T_prn)/2           # Шаг должен быть меньше ширины пика корреляции. Оптимально: 1/3 - 1/2 ширины пика
delta_T = (2/chip_rate)/2       # Это шаги в матрице из частот и времени

f_range = 10e3                  # пределы сдвига по доплеру
t_range = T_prn                 # ПСП повторяется с периодом 1 мс и кореляционный пик тоже

num_freq_points = int(2 * f_range / delta_F)    # Кол-во точек по оси частот
num_time_points = int(t_range / delta_T)        # Кол-во точек по оси времени   

dopler_shift = 2e3
delay_shift = 2e-4

max_correlation = 0
best_doppler = 0
best_delay_chips = 0

t = np.arange(0, T_prn, 1/chip_rate)
frequency = 10e4

frequency_shifted = frequency + dopler_shift
time_shifted = t - delay_shift

original_signal = np.sin(2 * np.pi * frequency * t)
delayed_shifted_signal = np.sin(2 * np.pi * (frequency_shifted) * (time_shifted))
correlation_max_T = 0
correlation_max_F = 0
correlation = np.correlate(original_signal, delayed_shifted_signal, mode='valid')[0]

for i in range(num_freq_points):
    correlation = np.correlate(original_signal, delayed_shifted_signal, mode='valid')[0]
    if correlation > correlation_max_F:
        correlation_max_F = correlation
    frequency_shifted = frequency_shifted - delta_F
    for j in range(num_time_points):
        correlation = np.correlate(original_signal, delayed_shifted_signal, mode='valid')[0]
        if correlation > correlation_max_T:
            correlation_max_T = correlation
        time_shifted = time_shifted - delta_T

# Нужно написать функцию, вычисляющую кореляцию сигнала без задержки и частоты с сигналом с задержкой и частотой, но у которого отнимается частота и время в соответсвии с шагом.
# И еще чтобы это было красиво на 3д графике показано было










