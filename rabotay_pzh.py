import numpy as np
import matplotlib.pyplot as plt
import time

# Параметры сигнала

chip_rate = 511e3                               # Скорость передачи символов
code_length = 511                               # Длина ПСП
fcode2fcarr = 2                                 # Во сколько раз ПЧ больше скорости чипов
fcarr2fs = 10                                   # Во сколько раз частота дискретизации больше ПЧ
samples_per_chip = fcode2fcarr * fcarr2fs       # Сколько отсчетов на один чип
fs = samples_per_chip * chip_rate               # Частота дискретизации
f_IF = chip_rate * fcode2fcarr                  # ПЧ
T_prn = 1e-3                                    # Период ПСП
accumulation_factor = 1                         # Сколько копим сигнала
T_coh = accumulation_factor * T_prn             # Время когерентного накопления

def generate_l1of():                            
    """
    Генератор ПСП L1OF: 9-разрядный LFSR с полиномом x^9 + x^5 + 1.
    Возвращает массив длиной code_length со значениями ±1.
    """
    register = [1] * 9                     # начальное состояние регистра
    prn_code = np.zeros(code_length)

    for i in range(code_length):
        prn_code[i] = register[6]          # выход (7-й разряд)
        feedback = register[8] ^ register[4]  # x^9 XOR x^5
        register = [feedback] + register[:-1]

    prn_code = 2 * prn_code - 1            # 0/1 -> ±1
    return prn_code

l1of_code = generate_l1of()
print(l1of_code[:9])                            # Убеждаемся, что Начальным символом в периоде ПС дальномерного кода является 1-ый символ в группе 111111100

l1of_code_oversampled = np.repeat(l1of_code, samples_per_chip)

delay_chips = np.random.randint(0, code_length)
delay_samples = delay_chips * samples_per_chip
l1of_code_oversampled_delayed = np.roll(l1of_code_oversampled, delay_samples)

# Временная ось
t = np.arange(len(l1of_code_oversampled_delayed)) / fs  # от 0 до 1 мс с шагом 1/fs

doppler_hz = np.random.uniform(-5000, 5000)
carrier = np.exp(1j * 2 * np.pi * (f_IF + doppler_hz) * t)
signal_complex = l1of_code_oversampled_delayed * carrier

cn0_db = 20
SNR_dB = cn0_db - 10*np.log10(chip_rate)

# def add_awgn_with_cn0(signal, cn0_db, fs):
    
#     # Нормализуем сигнал
#     signal_norm = signal / np.sqrt(np.mean(np.abs(signal)**2))
    
#     # Вычисляем СКО шума
#     cn0_linear = 10**(cn0_db / 10)
#     N0 = 1.0 / cn0_linear
#     noise_std = np.sqrt(N0 * fs / 2)
    
#     # Шум
#     noise = np.random.normal(0, noise_std, len(signal_norm)) + 1j * np.random.normal(0, noise_std, len(signal_norm))
    
#     return signal_norm + noise

def add_awgn_with_cn0(signal, cn0_db, fs):

    # Вычисляем реальную мощность сигнала
    C = np.mean(np.abs(signal)**2)
    
    cn0_linear = 10**(cn0_db / 10)
    N0 = C / cn0_linear 
    noise_std = np.sqrt(N0 * fs / 2)
    
    noise = np.random.normal(0, noise_std, len(signal)) + 1j * np.random.normal(0, noise_std, len(signal))
    
    return signal + noise, noise 

# Добавляем шум
signal_noisy, _ = add_awgn_with_cn0(signal_complex, cn0_db, fs)

# n_chips = 20  # покажем 20 чипов
# n_samples = n_chips * samples_per_chip  # 400 отсчетов

# plt.figure(figsize=(12, 4))
# plt.plot(np.abs(signal_noisy[:n_samples]), 'g-', linewidth=1)
# plt.title(f'Модуль (огибающая) зашумленного сигнала\nПервые {n_chips} чипов ({n_samples} отсчетов)')
# plt.xlabel('Номер отсчета')
# plt.ylabel('|Амплитуда|')
# plt.grid(True, alpha=0.3)

# # Границы чипов
# for chip in range(0, n_chips + 1):
#     plt.axvline(x=chip * samples_per_chip, color='gray', linestyle=':', alpha=0.5)

# plt.tight_layout()
# plt.show()



# plt.figure(figsize=(12, 4))
# plt.plot((l1of_code_oversampled[:n_samples]), 'g-', linewidth=1)
# plt.title(f'Модуль (огибающая) зашумленного сигнала\nПервые {n_chips} чипов ({n_samples} отсчетов)')
# plt.xlabel('Номер отсчета')
# plt.ylabel('|Амплитуда|')
# plt.grid(True, alpha=0.3)

# # Границы чипов
# for chip in range(0, n_chips + 1):
#     plt.axvline(x=chip * samples_per_chip, color='gray', linestyle=':', alpha=0.5)

# plt.tight_layout()
# plt.show()

# Диапазоны поиска
doppler_range = np.arange(-5000, 5001, 100)  # -5..+5 кГц, шаг 100 Гц
delay_range = np.arange(0, code_length*samples_per_chip, 2*samples_per_chip)  # задержки с шагом в чип (20 отсчетов)

# Последовательный поиск
def sequential_search(signal_noisy, l1of_code_oversampled, t):
    
    max_corr = 0
    best_doppler = 0
    best_delay = 0
    
    for doppler in doppler_range:  
        # Опорный сигнал с текущим доплером
        carrier = np.exp(1j * 2 * np.pi * (f_IF + doppler) * t)
        reference = l1of_code_oversampled * carrier
        
        for delay in delay_range:
            # Сдвигаем опорный
            reference_shifted = np.roll(reference, delay)
            
            # Корреляция
            corr = np.abs(np.dot(signal_noisy[:len(l1of_code_oversampled)], 
                               np.conj(reference_shifted)))
            
            if corr > max_corr:
                max_corr = corr
                best_doppler = doppler
                best_delay = delay
    
    return max_corr, best_doppler, best_delay

max_corr, best_doppler, best_delay = sequential_search(signal_noisy, l1of_code_oversampled, t)

print(f"Истинные: задержка {delay_chips} чипов, доплер {doppler_hz} Гц")
print(f"\nНайдено: задержка {best_delay/samples_per_chip} чипов, доплер {best_doppler} Гц")
print(max_corr)


# Параллельный поиск по доплеровскому смещению частоты
def parallel_search_doppler(signal_noisy, l1of_code_oversampled, t):
    max_corr = 0
    best_doppler = 0
    best_delay = 0

    carrier = np.exp(1j * 2 * np.pi * (f_IF) * t)
    reference = l1of_code_oversampled * carrier

    for delay in delay_range:
        reference_delay = np.roll(reference, delay)
        mixed_signal = reference_delay * signal_noisy
        spectrum_mixed_signal = np.fft.fft(mixed_signal)
        freq_axis = np.fft.fftfreq(len(spectrum_mixed_signal), d=1/fs)
        corr = np.max((np.abs(spectrum_mixed_signal))**2)
        max_idx = np.argmax((np.abs(spectrum_mixed_signal))**2)
        doppler_candidate = freq_axis[max_idx] - f_IF*2
        if corr > max_corr:
            max_corr = corr
            best_delay = delay
            best_doppler = doppler_candidate

    return max_corr, best_delay, best_doppler

max_corr, best_delay, best_doppler = parallel_search_doppler(signal_noisy, l1of_code_oversampled, t)
print('\nРезультат параллельного поиска по доплеровскому смещению частоты')
print(best_delay//samples_per_chip, best_doppler)
print(max_corr)


# Параллельный поиск по фазе кода
def fft_search_code_phase(signal_noisy, l1of_code_oversampled, t):

    max_corr = 0
    best_doppler = 0
    best_delay = 0
    
    carrier = np.exp(1j * 2 * np.pi * (f_IF) * t)
    reference = l1of_code_oversampled * carrier
    spectrum_reference_signal = np.fft.fft(reference)

    for doppler in doppler_range:
        signal_noisy_compensated = signal_noisy * np.exp(-1j * 2 * np.pi * doppler * t)
        spectrum_signal_noisy_compensated = np.fft.fft(signal_noisy_compensated)
        spectrum_mult = spectrum_signal_noisy_compensated * np.conj(spectrum_reference_signal)
        correlation_function = np.fft.ifft(spectrum_mult)
        corr = np.max((np.abs(correlation_function))**2)
        max_idx = np.argmax((np.abs(correlation_function))**2)
        if corr > max_corr:
            max_corr = corr
            best_doppler = doppler
            best_delay = max_idx
                
    return max_corr, best_doppler, best_delay

max_corr, best_doppler, best_delay = fft_search_code_phase(signal_noisy, l1of_code_oversampled, t)
print('\nРезультат параллельного поиска по фазе кода')
print(best_delay//20, best_doppler)
print(max_corr)



# def generate_noise_only(CN0_FOR_THRESHOLD, signal_length):

#     CN0_FOR_THRESHOLD = 10**(cn0_db / 10)
#     N0 = 1.0 / CN0_FOR_THRESHOLD
#     noise_std = np.sqrt(N0 * fs / 2)
    
#     noise = (np.random.normal(0, noise_std, signal_length) + 
#              1j * np.random.normal(0, noise_std, signal_length))
#     return noise

def generate_noise_only(cn0_db_param, signal_length):
   
    cn0_linear = 10**(cn0_db_param / 10) 
    N0 = 1.0 / cn0_linear
    noise_std = np.sqrt(N0 * fs / 2)
    
    noise = (np.random.normal(0, noise_std, signal_length) + 
             1j * np.random.normal(0, noise_std, signal_length))
    return noise



n_trials = 100  
p_fa_target = 0.01

CN0_FOR_THRESHOLD = 40  # дБГц

# 2. Длина сигнала (10220 отсчётов для 1 мс)
signal_length = 10220

# 3. Определение порога для БПФ поиска
max_correlations_fft = []

for i in range(n_trials):
    # Используем исправленную функцию
    noise = generate_noise_only(CN0_FOR_THRESHOLD, signal_length)
    
    max_corr, _, _ = fft_search_code_phase(noise, l1of_code_oversampled, t)
    max_correlations_fft.append(max_corr)
    
    if (i+1) % 20 == 0:
        print(f"Испытание {i+1}/{n_trials}")

max_correlations_fft = np.array(max_correlations_fft)
threshold_fft = np.percentile(max_correlations_fft, 90)

print(f"\nПорог для параллельного поиска по фазе кода: {threshold_fft:.4f}")
print(f"C/N₀ для порога: {CN0_FOR_THRESHOLD} дБГц")
print(f"Фактическая P_fa: {np.mean(max_correlations_fft > threshold_fft):.4f}")  

max_correlations = []

for i in range(n_trials):
    noise = generate_noise_only(CN0_FOR_THRESHOLD, signal_length)
    max_corr, _, _ = sequential_search(noise, l1of_code_oversampled, t)
    max_correlations.append(max_corr)
    print(f"Испытание {i+1}/{n_trials}")

max_correlations = np.array(max_correlations)
sequential_threshold = np.percentile(max_correlations, 99)

print(f"\nПорог для P_fa=1%: {sequential_threshold:.4f}")



# 1. Для БПФ поиска

# max_correlations_fft = []

# for i in range(n_trials):
#     noise = generate_noise_only()
#     max_corr, _, _= fft_search_code_phase(noise, l1of_code_oversampled, t)
    
#     max_correlations_fft.append(max_corr)

# max_correlations_fft = np.array(max_correlations_fft)
# threshold_fft = np.percentile(max_correlations_fft, 90)  # 99-й перцентиль

# print(f"\nПорог для параллельного поиска по фазе кода: {threshold_fft:.4f}")



# 2. Для параллельного по доплеру
max_correlations_dop = []

for i in range(n_trials):
    noise = generate_noise_only(CN0_FOR_THRESHOLD, signal_length)
    max_corr, _, _ = parallel_search_doppler(noise, l1of_code_oversampled, t)
    
    max_correlations_dop.append(max_corr)

    if (i+1) % 20 == 0:
        print(f"Испытание {i+1}/{n_trials}")

max_correlations_dop = np.array(max_correlations_dop)
threshold_dop = np.percentile(max_correlations_dop, 90)

print(f"\nПорог для параллельного поиска по доплеровскому смещению частоты: {threshold_dop:.4f}")





cn0_range_db = np.arange(37, 48, 1)
n_trials = 100

print("=== КРИВАЯ ОБНАРУЖЕНИЯ ===")
print(f"Порог: {threshold_fft:.2e}")
print(f"Испытаний на точку: {n_trials}\n")

probabilities = []

for cn0_db in cn0_range_db:
    detections = 0
    
    for trial in range(n_trials):
        # 1. Генерация кода
        l1of_code = generate_l1of()
        l1of_code_oversampled = np.repeat(l1of_code, samples_per_chip)
        
        # 2. Случайная задержка
        delay_chips = np.random.randint(0, code_length)
        delay_samples = delay_chips * samples_per_chip
        code_delayed = np.roll(l1of_code_oversampled, delay_samples)
        
        # 3. Временная ось
        t_local = np.arange(len(code_delayed)) / fs
        
        # 4. Случайный доплер
        doppler_hz = np.random.uniform(-5000, 5000)
        carrier = np.exp(1j * 2 * np.pi * (f_IF + doppler_hz) * t_local)
        
        # 5. Чистый сигнал
        signal_clean = code_delayed * carrier
        
        # 6. Добавление шума
        signal_noisy, _ = add_awgn_with_cn0(signal_clean, cn0_db, fs)
        
        # Поиск
        max_corr, _, _ = fft_search_code_phase(signal_noisy, l1of_code_oversampled, t_local)
        
        # Проверка порога
        if max_corr > threshold_fft:
            detections += 1
        
        if (trial + 1) % 10 == 0:
            print(f"  C/N₀={cn0_db}дБГц: {trial+1}/{n_trials}")
    
    p_d = 1-detections / n_trials
    probabilities.append(p_d)
    print(f"C/N₀={cn0_db}дБГц: P_d={p_d:.3f}")

# 3. Построение графика
plt.figure(figsize=(10, 6))
plt.plot(cn0_range_db, probabilities, 'bo-', linewidth=2, markersize=8)

plt.xlabel(r'$C/N_0$, дБГц', fontsize=12)
plt.ylabel(r'$P_d$', fontsize=12)
plt.title('Кривая обнаружения: параллельный поиск по фазе кода (T = 1 мс)', fontsize=14)
plt.grid(True, alpha=0.3)

plt.ylim([0, 1.05])
plt.xticks(cn0_range_db)
plt.yticks(np.arange(0, 1.1, 0.1))

plt.tight_layout()
plt.show()

print("\n=== РЕЗУЛЬТАТЫ ===")
for cn0, p_d in zip(cn0_range_db, probabilities):
    print(f"C/N₀={cn0}дБГц: P_d={p_d:.3f}")





cn0_range_db = np.arange(37, 48, 1)
n_trials = 100

print("=== КРИВАЯ ОБНАРУЖЕНИЯ ===")
print(f"Порог: {threshold_dop:.2e}")
print(f"Испытаний на точку: {n_trials}\n")

probabilities = []

for cn0_db in cn0_range_db:
    detections = 0
    
    for trial in range(n_trials):
        # 1. Генерация кода
        l1of_code = generate_l1of()
        l1of_code_oversampled = np.repeat(l1of_code, samples_per_chip)
        
        # 2. Случайная задержка
        delay_chips = np.random.randint(0, code_length)
        delay_samples = delay_chips * samples_per_chip
        code_delayed = np.roll(l1of_code_oversampled, delay_samples)
        
        # 3. Временная ось
        t_local = np.arange(len(code_delayed)) / fs
        
        # 4. Случайный доплер
        doppler_hz = np.random.uniform(-5000, 5000)
        carrier = np.exp(1j * 2 * np.pi * (f_IF + doppler_hz) * t_local)
        
        # 5. Чистый сигнал
        signal_clean = code_delayed * carrier
        
        # 6. Добавление шума
        signal_noisy, _ = add_awgn_with_cn0(signal_clean, cn0_db, fs)
        
        # Поиск
        max_corr, _, _ = parallel_search_doppler(signal_noisy, l1of_code_oversampled, t_local)
        
        # Проверка порога
        if max_corr > threshold_dop:
            detections += 1
        
        if (trial + 1) % 10 == 0:
            print(f"  C/N₀={cn0_db}дБГц: {trial+1}/{n_trials}")
    
    p_d = 1-detections / n_trials
    probabilities.append(p_d)
    print(f"C/N₀={cn0_db}дБГц: P_d={p_d:.3f}")

# 3. Построение графика
plt.figure(figsize=(10, 6))
plt.plot(cn0_range_db, probabilities, 'bo-', linewidth=2, markersize=8)

plt.xlabel(r'$C/N_0$, дБГц', fontsize=12)
plt.ylabel(r'$P_d$', fontsize=12)
plt.title('Кривая обнаружения: параллельный поиск по доплеру (T = 1 мс)', fontsize=14)
plt.grid(True, alpha=0.3)

plt.ylim([0, 1.05])
plt.xticks(cn0_range_db)
plt.yticks(np.arange(0, 1.1, 0.1))

plt.tight_layout()
plt.show()

print("\n=== РЕЗУЛЬТАТЫ ===")
for cn0, p_d in zip(cn0_range_db, probabilities):
    print(f"C/N₀={cn0}дБГц: P_d={p_d:.3f}")





cn0_range_db = np.arange(37, 48, 2)
n_trials = 30

print("=== КРИВАЯ ОБНАРУЖЕНИЯ ===")
print(f"Порог: {sequential_threshold:.2e}")
print(f"Испытаний на точку: {n_trials}\n")

probabilities = []

for cn0_db in cn0_range_db:
    detections = 0
    
    for trial in range(n_trials):
        # 1. Генерация кода
        l1of_code = generate_l1of()
        l1of_code_oversampled = np.repeat(l1of_code, samples_per_chip)
        
        # 2. Случайная задержка
        delay_chips = np.random.randint(0, code_length)
        delay_samples = delay_chips * samples_per_chip
        code_delayed = np.roll(l1of_code_oversampled, delay_samples)
        
        # 3. Временная ось
        t_local = np.arange(len(code_delayed)) / fs
        
        # 4. Случайный доплер
        doppler_hz = np.random.uniform(-5000, 5000)
        carrier = np.exp(1j * 2 * np.pi * (f_IF + doppler_hz) * t_local)
        
        # 5. Чистый сигнал
        signal_clean = code_delayed * carrier
        
        # 6. Добавление шума
        signal_noisy, _ = add_awgn_with_cn0(signal_clean, cn0_db, fs)
        
        # Поиск
        max_corr, _, _ = sequential_search(signal_noisy, l1of_code_oversampled, t_local)
        
        # Проверка порога
        if max_corr > sequential_threshold:
            detections += 1
        
        if (trial + 1) % 10 == 0:
            print(f"  C/N₀={cn0_db}дБГц: {trial+1}/{n_trials}")
    
    p_d = 1-detections / n_trials
    probabilities.append(p_d)
    print(f"C/N₀={cn0_db}дБГц: P_d={p_d:.3f}")

# 3. Построение графика
plt.figure(figsize=(10, 6))
plt.plot(cn0_range_db, probabilities, 'bo-', linewidth=2, markersize=8)

plt.xlabel(r'$C/N_0$, дБГц', fontsize=12)
plt.ylabel(r'$P_d$', fontsize=12)
plt.title('Кривая обнаружения: последовательный поиск (T = 1 мс)', fontsize=14)
plt.grid(True, alpha=0.3)

plt.ylim([0, 1.05])
plt.xticks(cn0_range_db)
plt.yticks(np.arange(0, 1.1, 0.1))

plt.tight_layout()
plt.show()

print("\n=== РЕЗУЛЬТАТЫ ===")
for cn0, p_d in zip(cn0_range_db, probabilities):
    print(f"C/N₀={cn0}дБГц: P_d={p_d:.3f}")
