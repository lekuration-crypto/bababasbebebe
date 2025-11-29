import numpy as np
import matplotlib.pyplot as plt
import time



# ============================================================
#                   ПАРАМЕТРЫ СИГНАЛА
# ============================================================

chip_rate = 511e3                           # Частота чипа L1OF, Гц
code_length = 511                           # Длина кода L1OF
T_prn = code_length / chip_rate             # Период ПСП, сек
fcode2fcarr = 2
fcarr2fs = 10
samples_per_chip = fcode2fcarr * fcarr2fs   # Количество отсчетов на один чип
fs = samples_per_chip * chip_rate           # Частота дискретизации, Гц (51.1 МГц)
f_IF = chip_rate * fcode2fcarr              # Промежуточная частота, Гц
accumulation_factor = 10                    # Сколько периодов кода повторять

SNR_dB = 20  # Отношение сигнал/шум в дБ



# ============================================================
#                       ГЕНЕРАТОР ПСП
# ============================================================

def generate_l1of():
    # Начальное состояние регистра
    register = [1]*9
    prn_code = np.zeros(code_length)

    for i in range(code_length):
        prn_code[i] = register[6]  # 7-я ячейка
        feedback = register[8] ^ register[4]  # x^9 XOR x^5
        register = [feedback] + register[:-1]

    prn_code = 2*prn_code - 1  # Преобразуем 0/1 в ±1
    return prn_code

# Генерируем ПСП
l1of_code = generate_l1of()
print("Первые 9 чипов ПСП:", l1of_code[:9])



# ============================================================
#                   Расширение по времени
# ============================================================

l1of_code_extended = np.tile(l1of_code, accumulation_factor)
extended_length = len(l1of_code_extended)
print(f"\nВремя накопления увеличено в {accumulation_factor} раз")
print(f"Новая длина сигнала: {extended_length} отсчетов")
print(f"Время накопления: {extended_length / chip_rate * 1000:.1f} мс")



# ============================================================
#                          Ресемплинг
# ============================================================

# Каждый чип повторяем samples_per_chip раз
prn_upsampled = np.repeat(l1of_code_extended, samples_per_chip)
print(f"Длина сигнала после апсемплинга: {len(prn_upsampled)} отсчетов")
print(f"Время сигнала: {len(prn_upsampled)/fs*1e3:.2f} мс")

# ============================================================
#                      МОДУЛЯЦИЯ на ПЧ
# ============================================================

# Создаем временную ось для сигнала в отсчетах
t_extended = np.arange(len(prn_upsampled)) / fs

# Модулируем сигнал на ПЧ
signal_IF = prn_upsampled * np.exp(-1j * 2 * np.pi * f_IF * t_extended)
print(f"Сигнал с ПЧ (f_IF={f_IF/1e6:.1f} МГц) создан, длина: {len(signal_IF)} отсчетов")



# ============================================================
#               Случайная задержка и доплер
# ============================================================

# Случайная задержка в отсчетах (0 до длины одного периода ПСП)
random_delay_samples = np.random.randint(0, code_length * samples_per_chip)

# Случайная частота Доплера (±5 кГц)
random_doppler = np.random.uniform(-5000, 5000)  # Гц

print(f"\nСлучайные параметры:")
print(f"Задержка: {random_delay_samples} отсчетов (~{random_delay_samples/chip_rate*1e6:.2f} мкс)")
print(f"Доплер: {random_doppler:.1f} Гц")

# Применяем задержку по времени к сигналу на ПЧ
delayed_signal = np.roll(signal_IF, random_delay_samples)

# Применяем сдвиг по частоте (доплер) к сигналу на ПЧ
signal_received = delayed_signal * np.exp(-1j * 2 * np.pi * random_doppler * t_extended)



# ============================================================
#                   Генерация AWGN шума
# ============================================================

def add_awgn(signal, snr_db):
    signal_power = np.mean(np.abs(signal)**2)        # Средняя мощность сигнала
    snr_linear = 10 ** (snr_db / 10)                 # SNR в линейный масштаб
    noise_power = signal_power / snr_linear          # Мощность шума
    sigma = np.sqrt(noise_power / 2)                 # СКО для комплексного шума

    noise = sigma * (np.random.randn(len(signal)) + 1j * np.random.randn(len(signal)))
    noisy_signal = signal + noise
    return noisy_signal, noise

# Добавляем шум к сигналу с ПЧ, задержкой и доплером
signal_noisy, noise_component = add_awgn(signal_received, SNR_dB)

print(f"Добавлен шум с SNR={SNR_dB} дБ, длина сигнала: {len(signal_noisy)} отсчетов")



# ============================================================
#                     Визуализация сигнала
# ============================================================

# Количество чипов для отображения ПСП
N_chips_plot = 20

# Количество отсчетов для отображения по времени для сигналов с ПЧ
N_plot = N_chips_plot * samples_per_chip
t_us = np.arange(N_plot) / fs * 1e6  # Время в мкс

plt.figure(figsize=(12, 10))

# ============================================================
#           График 1: Опорный ПСП (ступенчатый)
# ============================================================

plt.subplot(3, 1, 1)
time_steps = np.arange(N_chips_plot)
plt.scatter(time_steps, l1of_code_extended[:N_chips_plot], zorder=3, color='b')
plt.step(time_steps, l1of_code_extended[:N_chips_plot], where='post', color='b', label='Опорный ПСП')
plt.title("Опорный ПСП: первые 20 чипов")
plt.xlabel("Номер чипа")
plt.ylabel("Амплитуда")
plt.yticks([-1, 1], labels=["-1", "1"])
plt.grid(True)
plt.legend()

# ============================================================
#           График 2: Сигнал с ПЧ, задержкой и шумом
# ============================================================

plt.subplot(3, 1, 2)
plt.plot(t_us, np.real(signal_noisy[:N_plot]), 'r-', alpha=0.7, label='Сигнал с ПЧ + шум + задержка/доплер')
plt.title('Сигнал с ПЧ, шумом, задержкой и доплером')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.legend()

# ============================================================
#                   График 3: Только шум
# ============================================================

plt.subplot(3, 1, 3)
plt.plot(t_us, np.real(noise_component[:N_plot]), 'g-', label='Шум')
plt.title('Шумовая компонента')
plt.xlabel('Время (мкс)')
plt.ylabel('Амплитуда')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.show()



# ============================================================
#                     ПАРАМЕТРЫ ПОИСКА
# ============================================================

# Шаг по времени (в секундах)
delta_T_seconds = 0.5 / chip_rate
delta_T_samples = 0.5 * samples_per_chip # Берем сдвиг 0,5 чипа или же 10 отсчетов после ресемплинга

# Шаг по частоте (Гц)
delta_F = 50  # 50 Гц

# Диапазон по доплеру (±f_range)
f_range = 5000  # ±5 кГц

# Диапазон по времени (в секундах)
t_range_seconds = T_prn  # полный период ПСП

# Расчёт количества точек поиска
num_freq_points = int(2 * f_range / delta_F) + 1
num_time_points = int(t_range_seconds / delta_T_seconds) + 1

print("\nПараметры поиска:")
print(f"Длительность чипа: {1/chip_rate*1e6:.2f} мкс")
print(f"Длительность ПСП: {T_prn*1e3:.2f} мс")
print(f"Шаг по задержке: {delta_T_seconds*1e6:.2f} мкс")
print(f"Шаг по частоте: {delta_F} Гц")
print(f"Диапазон доплера: ±{f_range} Гц")
print(f"Диапазон задержек: 0 ... {t_range_seconds*1e3:.2f} мс")
print(f"Кол-во точек поиска: {num_freq_points} по частоте, {num_time_points} по времени")
print(f"Общее количество гипотез: {num_freq_points * num_time_points}")





# ============================================================
#                  ПОСЛЕДОВАТЕЛЬНЫЙ ПОИСК
# ============================================================

print('\nЗапуск последовательного поиска')

# Начало замера времени
start_time = time.time()

best_delay = 0
best_doppler = 0
best_correlation = -np.inf

signal_noisy_without_IF = signal_noisy * np.exp(1j * 2 * np.pi * f_IF * t_extended)

for i in range(-f_range, f_range, delta_F):

    signal_no_dopler = signal_noisy_without_IF * np.exp(1j * 2 * np.pi * i * t_extended)

    for j in range(0, code_length * samples_per_chip, int(delta_T_samples)):

        reference_signal_tau = np.roll(prn_upsampled, j)

        mixed_signal = signal_no_dopler * reference_signal_tau
        correlation = np.abs(np.sum(mixed_signal))

        if correlation > best_correlation:
            best_correlation = correlation
            best_delay = j
            best_doppler = i

# Конец замера времени
end_time = time.time()
print(f"Время выполнения поиска: {end_time - start_time:.3f} секунд")

print(f"Сгенерированная случайная задержка: {random_delay_samples} отсчетов ({random_delay_samples/samples_per_chip:.1f} чипов)")
print(f"Сгенерированный доплер: {random_doppler:.2f} Гц")
print(f"Найденная задержка: {best_delay} отсчетов ({best_delay/samples_per_chip:.1f} чипов)")
print(f"Найденное доплеровское смещение: {best_doppler:.2f} Гц")

# # ============================================================
# #                       2D визуализация
# # ============================================================
# plt.figure(figsize=(10,6))
# plt.imshow(corr_matrix, extent=[0, len(local_code), doppler_values[-1], doppler_values[0]],
#            aspect='auto', cmap='jet')
# plt.colorbar(label='Корреляция')
# plt.xlabel('Задержка (отсчеты)')
# plt.ylabel('Доплер (Гц)')
# plt.title('2D карта корреляции L1OF')
# plt.legend()
# plt.show()

# # ============================================================
# #                       3D визуализация
# # ============================================================
# X, Y = np.meshgrid(time_delays, doppler_values)
# Z = corr_matrix

# fig = plt.figure(figsize=(12,7))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X, Y, Z, cmap='jet')
# ax.set_xlabel('Задержка (отсчеты)')
# ax.set_ylabel('Доплер (Гц)')
# ax.set_zlabel('Корреляция')
# ax.set_title('3D карта корреляции L1OF')
# # Отметка максимума
# ax.scatter(best_delay, best_doppler, max_corr, color='red', s=50, label='Макс. корреляция')
# plt.show()



    






# ============================================================
#                 ПАРАЛЛЕЛЬНЫЙ ПОИСК ПО ДОПЛЕРУ
# ============================================================

print('\nЗапуск параллельного поиска по доплеру')

# Массивы для визуализации
delays = []
freqs_list = None         
correlation_map = []      

# Начало замера времени
start_time = time.time()

best_delay = 0
best_doppler = 0
best_correlation = -np.inf

print(f"Будем перебирать задержки: 0 до {code_length * samples_per_chip-1} с шагом {int(delta_T_samples)} отсчетов")
print(f"Количество гипотез по задержке: {int(code_length * samples_per_chip // delta_T_samples)}")

for i in range(0, code_length * samples_per_chip, int(delta_T_samples)):
    
    reference_signal_tau = np.roll(prn_upsampled, i)

    mixed_signal = signal_noisy_without_IF * reference_signal_tau

    spectrum_tau = np.fft.fft(mixed_signal)
    power_spectrum_tau = np.abs(spectrum_tau) ** 2

    freqs = np.fft.fftshift(np.fft.fftfreq(len(mixed_signal), d=1/fs))  # сдвинутая ось (Теперь по-человечески [-Fs/2, -Fs/2+Δf, ..., -Δf, 0, Δf, ..., Fs/2-Δf])
    power_spectrum_shifted = np.fft.fftshift(power_spectrum_tau)        # сдвинутый спектр
    doppler_mask = (np.abs(freqs) <= 5000)                              # Создаем маску для диапазона ±5 кГц
    power_spectrum_filtered = power_spectrum_shifted[doppler_mask]      # Применяем маску к спектру
    freqs_filtered = freqs[doppler_mask]

    if freqs_list is None:
        freqs_list = -freqs_filtered
    
    correlation_map.append(power_spectrum_filtered)
    delays.append(i)

    max_power_index = np.argmax(power_spectrum_filtered)
    max_power_value = power_spectrum_filtered[max_power_index]
    doppler_freq = freqs_filtered[max_power_index]

    if max_power_value > best_correlation:
        best_correlation = max_power_value
        best_delay = i
        best_doppler = -doppler_freq

# Конец замера времени
end_time = time.time()
print(f"Время выполнения поиска: {end_time - start_time:.3f} секунд")

print(f"Сгенерированная случайная задержка: {random_delay_samples} отсчетов ({random_delay_samples/samples_per_chip:.1f} чипов)")
print(f"Сгенерированный доплер: {random_doppler:.2f} Гц")
print(f"Найденная задержка: {best_delay} отсчетов ({best_delay/samples_per_chip:.1f} чипов)")
print(f"Найденное доплеровское смещение: {best_doppler:.2f} Гц")

correlation_map = np.array(correlation_map)  
delays = np.array(delays)
freqs_list = np.array(freqs_list)
correlation_map_normalized = correlation_map / np.max(correlation_map)

# ============================================================
#                       2D визуализация
# ============================================================

plt.figure(figsize=(10, 6))
plt.imshow(
    correlation_map_normalized,
    aspect='auto',
    extent=[freqs_list[0], freqs_list[-1], delays[-1], delays[0]],
    cmap='jet'
)
plt.colorbar(label='Correlation Power')
plt.xlabel("Doppler frequency (Hz)")
plt.ylabel("Code delay (samples)")
plt.title("2D Doppler–Delay Search Map")
plt.show()

# ============================================================
#                       3D визуализация
# ============================================================

freq_grid, delay_grid = np.meshgrid(freqs_list, delays)

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(freq_grid, delay_grid, correlation_map_normalized, cmap='jet')

ax.set_xlabel("Doppler frequency (Hz)")
ax.set_ylabel("Code delay (samples)")
ax.set_zlabel("Correlation power")
ax.set_title("3D Doppler–Delay Surface")
plt.show()





# ============================================================
#                ПАРАЛЛЕЛЬНЫЙ ПОИСК ПО ФАЗЕ КОДА
# ============================================================

print('\nЗапуск параллельного поиска по фазе кода')

dopplers = []
delays = []
corr_map = []

# Начало замера времени
start_time = time.time()

best_delay = 0
best_doppler = 0
best_correlation = -np.inf

print(f"Будем перебирать частоту: -5 кГц до 5 кГц с шагом {delta_F} Гц")
print(f"Количество гипотез по частоте: {num_freq_points}")

for i in range(-f_range, f_range, delta_F):
    
    signal_no_dopler = signal_noisy_without_IF * np.exp(1j * 2 * np.pi * i * t_extended)

    spectrum_noisy_signal_no_dopler = np.fft.fft(signal_no_dopler)                                    # БПФ входного сигнала (после компенсации доплера)
    spectrum_reference_signal = np.fft.fft(prn_upsampled)                                             # БПФ опорного кода
    spectrum_multiplication = spectrum_noisy_signal_no_dopler * np.conj(spectrum_reference_signal)    # Комплексное сопряжение и умножение

    correlation_function = np.fft.ifft(spectrum_multiplication)     # Обратное БПФ для перехода во временную область

    correlation_power = np.abs(correlation_function) ** 2           # Поиск максимума в корреляционной функции

    search_range = code_length * samples_per_chip
    correlation_power_limited = correlation_power[:search_range]    # Берем только первую часть корреляционной функции (один период)

    corr_map.append(correlation_power_limited)
    dopplers.append(-i)

    max_correlation = np.max(correlation_power_limited)             # Находим максимальное значение корреляции
    delay_t = np.argmax(correlation_power_limited)                  # Находим индекс, где достигнут максимум

    if max_correlation > best_correlation:
        best_correlation = max_correlation
        best_delay = delay_t
        best_doppler = i

# Конец замера времени
end_time = time.time()
print(f"Время выполнения поиска: {end_time - start_time:.3f} секунд")

print(f"Сгенерированная случайная задержка: {random_delay_samples} отсчетов ({random_delay_samples/samples_per_chip:.1f} чипов)")
print(f"Сгенерированный доплер: {random_doppler:.2f} Гц")
print(f"Найденная задержка: {best_delay} отсчетов ({best_delay/samples_per_chip:.1f} чипов)")
print(f"Найденное доплеровское смещение: {best_doppler:.2f} Гц")

# Превращаем список в матрицу
corr_map = np.array(corr_map)  
corr_map_normalized = corr_map / np.max(corr_map)

# ============================================================
#                       2D визуализация
# ============================================================

plt.figure(figsize=(10, 5))
plt.imshow(
    corr_map_normalized,
    extent=[0, search_range, dopplers[0], dopplers[-1]],
    aspect='auto',
    cmap='viridis'
)
plt.colorbar(label="Correlation Power")
plt.xlabel("Code phase (samples)")
plt.ylabel("Doppler (Hz)")
plt.title("2D Correlation Map (Time vs Doppler)")
plt.show()

# ============================================================
#                       3D визуализация
# ============================================================

X, Y = np.meshgrid(np.arange(search_range), dopplers)
Z = corr_map_normalized

fig = plt.figure(figsize=(12, 7))
ax = fig.add_subplot(111, projection='3d')

ax.plot_surface(X, Y, Z, cmap='viridis')
ax.set_xlabel("Code phase (samples)")
ax.set_ylabel("Doppler (Hz)")
ax.set_zlabel("Correlation Power")
ax.set_title("3D Correlation Surface")
plt.show()