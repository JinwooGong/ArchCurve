import numpy as np
import matplotlib.pyplot as plt
from skimage.morphology import skeletonize
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path
from scipy.interpolate import splprep, splev


### Apply gamma
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
        
    return np.array(corrected)

#### Binarization
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
    
    mask = invert_binary_image(mask)
    return mask.astype(np.uint8)

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

## Seg arch curve
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

### Apply Gaussian Filter

import numpy as np
from scipy.ndimage import gaussian_filter

def detect_edges(binary_image, edge_width=2):
    """
    이진 이미지에서 가장자리를 감지합니다.
    
    Args:
        binary_image (np.ndarray): 0과 1로 이루어진 2차원 이진 이미지
        edge_width (int): 가장자리로 검출할 픽셀의 너비
    
    Returns:
        np.ndarray: 가장자리 픽셀이 1인 이진 이미지
    """
    # 패딩된 이미지 생성
    padded = np.pad(binary_image, pad_width=edge_width, mode='constant', constant_values=0)
    edges = np.zeros_like(binary_image)
    
    # 각 픽셀의 주변을 확인하여 가장자리 감지
    for i in range(edge_width, padded.shape[0]-edge_width):
        for j in range(edge_width, padded.shape[1]-edge_width):
            if padded[i, j] == 1:
                window = padded[i-edge_width:i+edge_width+1, j-edge_width:j+edge_width+1]
                if not np.all(window == 1):
                    edges[i-edge_width, j-edge_width] = 1
                    
    return edges

def smooth_edges(binary_image, sigma=2.0, edge_width=2, iterations=2):
    """
    이진 이미지의 가장자리를 가우시안 필터로 부드럽게 합니다.
    
    Args:
        binary_image (np.ndarray): 0과 1로 이루어진 2차원 이진 이미지
        sigma (float): 가우시안 필터의 표준편차
        edge_width (int): 가장자리 검출 너비
        iterations (int): 가우시안 필터 적용 반복 횟수
        
    Returns:
        np.ndarray: 가장자리가 부드러워진 이미지
    """
    # 원본 이미지 복사
    smoothed = binary_image.copy().astype(float)
    
    # 가장자리 감지
    edges = detect_edges(binary_image, edge_width)
    
    # 반복적으로 가우시안 필터 적용
    edge_smoothed = edges.astype(float)
    for _ in range(iterations):
        edge_smoothed = gaussian_filter(edge_smoothed, sigma=sigma)
    
    # 원본 이미지에서 가장자리 부분만 부드럽게 처리
    mask = edges == 1
    smoothed[mask] = edge_smoothed[mask]
    
    # 값의 범위를 0~1로 정규화
    if np.max(smoothed) > 1:
        smoothed = smoothed / np.max(smoothed)
    smoothed[smoothed != 1] = 0
    return smoothed

### Extract skeleton
def extract_skeleton(image):
    """
    이미지에서 골격을 추출합니다.
    
    Args:
        image: 2D 이미지
    Returns:
        np.ndarray: 골격화된 이미지
    """
    # 골격화 수행
    skeleton = skeletonize(image, method='lee')
    skeleton[skeleton == 255] = 1
    return skeleton

### Find Longest Path
def find_longest_path(binary_image):
    """
    미로와 같은 이진 이미지에서 가장 긴 경로를 찾습니다.
    
    Args:
        binary_image (np.ndarray): 0과 1로 이루어진 2차원 이진 이미지
        
    Returns:
        np.ndarray: 가장 긴 경로만 표시된 이진 이미지
    """
    # 이미지의 모든 픽셀 위치를 노드로 변환
    y_indices, x_indices = np.where(binary_image == 1)
    points = list(zip(y_indices, x_indices))
    n_points = len(points)
    
    if n_points == 0:
        return np.zeros_like(binary_image)
    
    # 포인트 인덱스 매핑 생성
    point_to_idx = {point: idx for idx, point in enumerate(points)}
    
    # 인접 행렬 생성
    adjacency_matrix = np.zeros((n_points, n_points))
    
    # 8방향 이웃 검사
    neighbors = [(-1,-1), (-1,0), (-1,1), (0,-1), (0,1), (1,-1), (1,0), (1,1)]
    
    for i, (y, x) in enumerate(points):
        for dy, dx in neighbors:
            ny, nx = y + dy, x + dx
            neighbor = (ny, nx)
            if neighbor in point_to_idx:
                j = point_to_idx[neighbor]
                adjacency_matrix[i, j] = 1
    
    # 희소 행렬로 변환
    graph = csr_matrix(adjacency_matrix)
    
    # 모든 점 쌍 사이의 최단 거리 계산
    distances = shortest_path(graph)
    
    # 가장 긴 경로의 시작점과 끝점 찾기
    max_distance = 0
    start_idx = end_idx = 0
    
    for i in range(n_points):
        for j in range(i + 1, n_points):
            if distances[i, j] > max_distance and distances[i, j] != np.inf:
                max_distance = distances[i, j]
                start_idx = i
                end_idx = j
    
    # 최장 경로 추적
    result = np.zeros_like(binary_image)
    
    def get_path(start_idx, end_idx):
        """시작점과 끝점 사이의 경로를 찾습니다."""
        current = start_idx
        path = [current]
        while current != end_idx:
            neighbors = np.where(adjacency_matrix[current] > 0)[0]
            next_point = min(neighbors, 
                           key=lambda x: distances[x, end_idx] if distances[x, end_idx] != np.inf else float('inf'))
            path.append(next_point)
            current = next_point
        return path
    
    # 경로를 이미지에 표시
    path_indices = get_path(start_idx, end_idx)
    for idx in path_indices:
        y, x = points[idx]
        result[y, x] = 1
    
    return result

def sort_points_by_distance(points):
    """
    점들을 거리 기반으로 정렬하여 선의 순서를 만듭니다.
    """
    if len(points) == 0:
        return points
        
    # 시작점 (가장 왼쪽 점)
    start_idx = np.argmin(points[:, 0])
    ordered = [points[start_idx]]
    remaining = list(range(len(points)))
    remaining.remove(start_idx)
    
    # 가장 가까운 점을 찾아가며 정렬
    while remaining:
        last_point = ordered[-1]
        distances = np.sqrt(np.sum((points[remaining] - last_point) ** 2, axis=1))
        nearest_idx = remaining[np.argmin(distances)]
        ordered.append(points[nearest_idx])
        remaining.remove(nearest_idx)
    
    return np.array(ordered)

def smooth_binary_line(curve_image, smoothness=0.3, num_points=11):
    """
    이진 이미지의 선을 부드럽게 만듭니다.
    
    Args:
        binary_image (np.ndarray): 0과 1로 이루어진 2차원 이진 이미지
        smoothness (float): 부드러움 정도 (0에 가까울수록 원본과 유사)
        num_points (int): 부드러운 곡선의 점 개수
    
    Returns:
        tuple: (부드러운 x좌표 배열, 부드러운 y좌표 배열)
    """
    y_coords, x_coords = np.where(curve_image == 1)
    # 좌표점들을 순서대로 정렬
    points = np.column_stack((x_coords, y_coords))
    # 점들을 연결 순서대로 정렬
    points = sort_points_by_distance(points)
    
    if len(points) < 4:
        return points[:, 0], points[:, 1]
    
    try:
        # 스플라인 보간
        tck, u = splprep([points[:, 0], points[:, 1]], s=smoothness, k=3)
        u_new = np.linspace(0, 1, num_points)
        x_smooth, y_smooth = splev(u_new, tck)
        
        return x_smooth, y_smooth
    except:
        # 보간이 실패할 경우 원본 반환
        return points[:, 0], points[:, 1]
    
def visualize_results(binary_image, x_smooth, y_smooth):
    # 한글 폰트 설정
    plt.rcParams['font.family'] = 'Malgun Gothic'  # 나눔고딕 폰트 사용
    plt.rcParams['axes.unicode_minus'] = False    # 마이너스 기호 깨짐 방지
    """
    원본 이진 이미지와 부드러워진 선을 시각화합니다.
    """
    plt.figure(figsize=(10, 10))
    
    # 원본 이미지
    plt.subplot(121)
    plt.imshow(binary_image, cmap='gray')
    plt.title('원본 이미지')
    plt.axis('image')
    
    # 부드러워진 선
    plt.subplot(122)
    plt.imshow(binary_image, cmap='gray', alpha=0.3)
    plt.plot(x_smooth, y_smooth, 'r-', linewidth=2)
    plt.title('부드러워진 선')
    plt.axis('image')
    
    plt.tight_layout()
    plt.show()