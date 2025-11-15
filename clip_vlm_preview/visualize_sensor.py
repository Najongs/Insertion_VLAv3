import numpy as np
import matplotlib.pyplot as plt
import sys
import os

def visualize_sensor_data(npz_file):
    """
    .npz 파일에 저장된 센서 데이터를 시각화합니다.

    - force 데이터는 선 그래프로 표시합니다.
    - aline (fpi) 데이터는 2D 이미지로 표시합니다.
    """
    try:
        # 데이터 로드
        data_npz = np.load(npz_file)
    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다: {npz_file}")
        return

    # 'forces'와 'alines' 키 확인
    if 'forces' not in data_npz or 'alines' not in data_npz:
        print(f"오류: '{npz_file}' 파일에 'forces' 또는 'alines' 키가 없습니다.")
        print(f"사용 가능한 키: {list(data_npz.keys())}")
        return

    force_data = data_npz['forces']
    aline_data = data_npz['alines']
    print(f"Force 데이터를 로드했습니다. 형태: {force_data.shape}")
    print(f"Aline 데이터를 로드했습니다. 형태: {aline_data.shape}")

    # 시각화
    fig, axs = plt.subplots(2, 1, figsize=(12, 10), gridspec_kw={'height_ratios': [1, 3]})
    
    # 파일명을 제목에 추가
    base_filename = os.path.basename(npz_file)
    fig.suptitle(f'Sensor Data Visualization\n({base_filename})', fontsize=16)

    # Force 데이터 플롯
    axs[0].plot(force_data)
    axs[0].set_title('Force Data')
    axs[0].set_xlabel('Time step')
    axs[0].set_ylabel('Force')
    axs[0].grid(True)

    # Aline (FPI) 데이터 플롯 (이미지)
    im = axs[1].imshow(aline_data.T, aspect='auto', cmap='gray', interpolation='none')
    axs[1].set_title('Aline (FPI) Data')
    axs[1].set_xlabel('Time step')
    axs[1].set_ylabel('FPI sensor index')
    fig.colorbar(im, ax=axs[1], label='Sensor Reading')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # 이미지 파일로 저장
    output_filename = 'sensor_data_visualization.png'
    plt.savefig(output_filename)
    print(f"시각화 결과를 '{output_filename}' 파일로 저장했습니다.")
    plt.close()

if __name__ == "__main__":
    if len(sys.argv) > 1:
        npz_file_path = sys.argv[1]
        visualize_sensor_data(npz_file_path)
    else:
        print("사용법: python visualize_sensor.py <.npz 파일 경로>")
