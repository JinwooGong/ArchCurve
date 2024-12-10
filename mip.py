import numpy as np
from scipy import ndimage
from typing import Tuple, Set
from utils.file_manager import load_scan, windowing_image

def gamma_correction(image, gamma, c=1.0):
    """
    감마 변환을 수행하는 함수
    
    Parameters:
    -----------
    image : numpy.ndarray
        입력 이미지 (0-255 범위의 uint8 또는 0-1 범위의 float)
    gamma : float
        감마 값 (γ)
        γ > 1: 어두운 영역 압축 (밝은 부분 강조)
        γ < 1: 밝은 영역 압축 (어두운 부분 강조)
    c : float, optional
        스케일링 팩터 (기본값: 1.0)
    
    Returns:
    --------
    numpy.ndarray
        감마 변환이 적용된 이미지
    """
    # 입력 이미지를 0-1 범위로 정규화
    if image.dtype == np.uint8:
        image = image / 255.0
        
    # 감마 변환 수행: s = c * r^γ
    corrected = c * np.power(image, gamma)
    
    # 값의 범위를 0-1로 클리핑
    corrected = np.clip(corrected, 0, 1)
    
    # 원본이 uint8이었다면 다시 0-255로 변환
    if image.dtype == np.uint8:
        corrected = (corrected * 255).astype(np.uint8)
        
    return corrected
def region_growing_binary(image, seed_points=None, threshold=0.1):
    """
    Region growing 방식의 이진화
    
    Parameters:
    -----------
    image : numpy.ndarray
        입력 이미지
    seed_points : list of tuple
        시작점 좌표 [(y1,x1), (y2,x2), ...]
    threshold : float
        성장 기준 임계값
        
    Returns:
    --------
    numpy.ndarray
        이진화된 이미지
    """
    if seed_points is None:
        # 이미지 중심을 시작점으로 사용
        center_y, center_x = [dim//2 for dim in image.shape]
        seed_points = [(center_y, center_x)]
    
    # 방문 마스크 초기화
    mask = np.zeros_like(image, dtype=np.uint8)
    
    # 시작점의 값을 기준으로 설정
    for seed_y, seed_x in seed_points:
        reference = image[seed_y, seed_x]
        stack = [(seed_y, seed_x)]
        
        while stack:
            y, x = stack.pop()
            if not mask[y, x]:
                # 현재 픽셀 방문 표시
                mask[y, x] = 1
                
                # 4방향 이웃 픽셀 확인
                for dy, dx in [(-1,0), (1,0), (0,-1), (0,1)]:
                    new_y, new_x = y + dy, x + dx
                    
                    # 이미지 범위 체크
                    if (0 <= new_y < image.shape[0] and 
                        0 <= new_x < image.shape[1] and 
                        not mask[new_y, new_x]):
                        # 임계값 이내의 차이를 보이는 픽셀 추가
                        if abs(image[new_y, new_x] - reference) < threshold:
                            stack.append((new_y, new_x))
    
    return mask.astype(np.uint8)

def post_process_binary(binary_img, min_size=100):
    """
    이진화 이미지의 후처리
    
    Parameters:
    -----------
    binary_img : numpy.ndarray
        이진화된 이미지
    min_size : int
        최소 객체 크기
        
    Returns:
    --------
    numpy.ndarray
        후처리된 이진화 이미지
    """
    # 작은 객체 제거
    cleaned = ndimage.binary_opening(binary_img, structure=np.ones((3,3)))
    
    # 레이블링
    labeled, num_features = ndimage.label(cleaned)
    
    # 크기가 작은 객체 제거
    for i in range(1, num_features + 1):
        if np.sum(labeled == i) < min_size:
            cleaned[labeled == i] = 0
            
    return cleaned.astype(np.uint8)

def find_largest_component(grid):
    """
    2차원 리스트로 표현된 이진 이미지에서 가장 큰 연결된 1의 영역을 찾습니다.
    반복문을 사용하여 재귀 제한을 피합니다.
    
    Args:
        grid: 2차원 리스트 (각 셀은 0 또는 1)
    
    Returns:
        new_grid: 가장 큰 영역만 1로 표시된 2차원 리스트
        max_size: 가장 큰 영역의 크기
    """
    
    rows = len(grid)
    cols = len(grid[0])
    visited = set()
    
    def get_component(start_r, start_c):
        """반복문을 사용하여 연결된 모든 1을 찾아 반환"""
        if grid[start_r][start_c] != 1:
            return set()
            
        component = set()
        stack = [(start_r, start_c)]
        
        while stack:
            r, c = stack.pop()
            
            if (r, c) in visited:
                continue
                
            if r < 0 or r >= rows or c < 0 or c >= cols:
                continue
                
            if grid[r][c] != 1:
                continue
                
            visited.add((r, c))
            component.add((r, c))
            
            # 8방향의 이웃을 스택에 추가
            neighbors = [
                (r+1,c), (r-1,c), (r,c+1), (r,c-1),
                (r+1,c+1), (r-1,c-1), (r+1,c-1), (r-1,c+1)
            ]
            
            for nr, nc in neighbors:
                if (nr, nc) not in visited:
                    stack.append((nr, nc))
            
        return component
    
    # 모든 연결 영역 찾기
    all_components = []
    for i in range(rows):
        for j in range(cols):
            if (i, j) not in visited and grid[i][j] == 1:
                component = get_component(i, j)
                if component:
                    all_components.append(component)
    
    # 빈 그리드 처리
    if not all_components:
        return [[0 for _ in range(cols)] for _ in range(rows)], 0
    
    # 가장 큰 영역 찾기
    largest = max(all_components, key=len)
    
    # 결과 그리드 생성
    result = [[0 for _ in range(cols)] for _ in range(rows)]
    for r, c in largest:
        result[r][c] = 1
    
    return result, len(largest)

def invert_binary_image(image):
    """
    이진 이미지(0과 1로 구성된 2차원 리스트)의 값을 반전시킵니다.
    
    Args:
        image: 2차원 리스트 형태의 이진 이미지
        
    Returns:
        inverted_image: 반전된 이진 이미지
    """
        
    rows = len(image)
    cols = len(image[0])
    
    # 새로운 이미지 생성
    inverted = [[0 for _ in range(cols)] for _ in range(rows)]
    
    # 각 픽셀 반전
    for i in range(rows):
        for j in range(cols):
            inverted[i][j] = 1 if image[i][j] == 0 else 0
            
    return np.array(inverted)

if __name__ == "__main__":
    dcm_file = "D:/2차년도/골이식재양/고대구로/덴티움/00749625 정순덕/술전"
    slices, original_data, properties = load_scan(dcm_file)

    tr_data = np.transpose(original_data, (2,1,0))
    top = tr_data.shape[0]
    new_data = tr_data[top-200:top-100]
    
    print(f"Window Center: {properties['window_center']}")
    print(f"Window Width: {properties['window_width']}")
    
    mip_image = np.max(new_data, axis=0)
    
    # windowing and normalize
    window_image = windowing_image(mip_image, properties, use_normalize=True)
    # bone_image = windowing_image(mip_image, properties,window=(400,1800), use_normalize=True)
    
    # Apply gamma
    test_image = gamma_correction(window_image, gamma=1.5)
    
    midle_point = (int(test_image.shape[0] / 2), int(test_image.shape[1] / 2))
    seed_points = ([(0,0), midle_point])
    binary_region = region_growing_binary(test_image, seed_points=seed_points, threshold=0.1)
    binary_region = invert_binary_image(binary_region)
    
    result, max_len = find_largest_component(binary_region)
    