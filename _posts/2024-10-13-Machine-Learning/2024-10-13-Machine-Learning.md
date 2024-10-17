---
layout: post
title: Machine Learning cơ bản - phần 1
image: /images/2024-10-13-Machine-Learning
---

# Giới thiệu về bài toán Linear Regression
Ta có một loạt các cặp dữ liệu thể hiện mối quan hệ giữa **diện tích đất** và **giá đất** ở một thành phố như sau. Nếu như mình hỏi bạn nếu như có một bãi đất với diện tích nào đó mà không có trong bản dữ liệu thì bạn có thể **đoán** được giá của bãi đất đó không? Hơi khó đúng không?

Nhưng nếu như ta mô tả đám dữ liệu đó dưới dạng các điểm trên một **hệ trục tọa độ** như thế này thì sao?
![Ảnh 1]({{page.image}}/anh-1.png)

Không phải ta sẽ dễ dàng có thể dự đoán được nó hay sao? Mọi thứ sẽ trở nên đơn giản hơn nếu như ta biết cách biểu diễn dữ liệu một cách trực quan hơn.

Nhưng đó là với bạn, một cỗ máy sinh học chạy bằng cơm. Còn với máy tính thì sao? Nếu bạn cho nó một bộ giữ liệu như vậy và yêu cầu nó dự đoán giá nhà với diện tích bất kì thì sao? Câu hỏi tưởng dễ nhưng giờ lại bắt đầu đau não rồi đây.

## Nhận xét về bài toán
Quay trở lại với cái đường thẳng này. Ta có các nhận xét như sau:
* Các điểm đã cho nằm khá gần đường thẳng. 
* Vì là 1 đường thẳng lên sẽ có phương trình là $y_i = ax_i + b$.

Giả sử ta có được đường thẳng được gọi tốt nhất thì vì sao nó lại là tốt nhất. Kiểu chúng ta phải có một thứ gì đó có thể đánh giá được một cái gì đó hơn một cái gì đó khác chứ.

Ta gọi $x_i$ là diện tích của căn nhà và $y_i$ là giá tiền của căn nhà đó.

Nếu đường thẳng là tốt thì $\hat{y_i} = f(x_i)$ phải xấp xỉ với $y_i$. 

Và thứ gì đó ở đây giúp ta đánh giá được độ tệ hại của một đường thẳng với một điểm là:

$$
    (\hat{y_i} - y_i)^2
$$

Và với $n$ điểm là:

$$
    \sum_{i = 1}^{n}(\hat{y_i} - y_i)^2
$$

***Ai đó:** Từ từ, cái con mẹ gì đây, tại sao lại là bình phương? Ta có thể dùng giá trị tuyệt đối mà.*

Lý do chúng ta bình phương là nhằm để cho phương trình không âm và giá trị nhỏ nhất của hàm cũng là $0$ giúp ta dễ tính toán. Ngoài ra nó còn giúp tăng độ khuếch đại với các điểm ở càng xa đừng 
thẳng (Mình sẽ có một bài viết để phân tích kĩ hơn về nó).

Giờ tay hãy phân tích công thức trên một tí.

$$
    L(a, b) = \sum_{i = 1}^{n}(\hat{y_i} - y_i)^2 \\
    L(a, b) = \sum_{i = 1}^{n}(f(x_i) - y_i)^2 \\
    L(a, b) = \sum_{i = 1}^{n}(ax_i + b - y_i)^2 \\
$$

Nếu như ta có một hàm $f$ có thể dự đoán chính xác với $n$ điểm đã cho thì hàm $L$ của ta sẽ đạt giá trị nhỏ nhất. Vậy bài toán lúc này sẽ đưa về việc tìm cặp $(a, b)$ sao cho hàm $L$ đạt giá trị nhỏ nhất. Việc ta đi tìm cặp $(a, b)$ sao cho hàm $L$ đạt giá trị nhỏ nhất được gọi là **train**.

# Tìm giá trị nhỏ nhất của hàm $L$
Có rất nhiều cách khác nhau để tìm ra cặp $(a,b)$, ở đây mình sẽ liệt kê một bài cách "thú vị". À, trước khi đọc tiếp thì mình sẽ thay đổi một chút về phần ký hiệu để tiện hơn cho sau này.

$$
    w_1 = a \\
    w_2 = b
$$

## Bằng đại số tuyến tính
Như cái tên, thì muốn hiểu phần này bạn phải biết một chút kiến thứ về **đại số tuyến tính**. Mình cũng sẽ viết một bài để giới thiệu về các đại số tuyến tính cơ bản để dùng trong Machine Learning.

Ta có thể biểu diễn cách tham số và biến ở phần trên lại dưới dạng ma trận như sau:

$$
    W = \begin{bmatrix}
    w_1 \\ w_2
    \end{bmatrix} 
$$

$$
    X_i = \begin{bmatrix}
    x_1 & 1
    \end{bmatrix}
$$

$$
    X = \begin{bmatrix}
    x_1 & 1 \\
    x_2 & 1 \\
    x_3 & 1 \\
    ... & .. \\
    x_n & 1
    \end{bmatrix}
$$


$$
    Y = \begin{bmatrix}
    y_1 \\ y_2 \\ y_3 \\ ... \\ y_n
    \end{bmatrix}
$$

Tiếp theo ta sẽ biểu diễn hàm hàm $f$ và hàm $L$ dưới dạng ma trận:

$$
    \hat{Y} = f(X) = XW \\
    \begin{align}
        L(W) = ||Y - \hat{Y} ||_2^2 \\
        L(W) = ||Y - XW ||_2^2 \\
    \end{align}
$$

Với $\|\|z\|\|_2$ là **Euclidean norm** hay **khoảng cách Euclid** của $z$ hay cho dễ hiểu hơn là tổng bình phương của các phần tử trong $z$.

Giờ mới là ma thuật này.

Như ở cấp 3 có học thì **cực trị** của một hàm số sẽ nằm ở những điểm có **đạo hàm** bằng $0$ hoặc không tồn tại đạo hàm tại điểm đó. Ở đây phương trình là **phương trình bậc 2** nên sẽ luôn có tồn tại đạo hàm, không như việc sử dụng **giá trị tuyệt đối**. Thì ta có công thức đạo hàm của hàm $L$ theo biến $W$.

$$
    \frac{\partial{L}}{\partial{W}}
    = X^T(XW - Y)
$$

Bạn có thể đọc qua [tài liệu này](https://ccrma.stanford.edu/~dattorro/matrixcalc.pdf) để tìm hiểu thêm và **đạo hàm của ma trận**.

Quay trở lại bài toán, thì nếu như đạo hàm bằng $0$ tức là:

$$
    \frac{\partial{L}}{\partial{W}} = 0\\
    \longrightarrow X^T(XW - Y) = 0 \\
    \longrightarrow X^T X W = X^T Y ~~~ (1)
$$

Trong trường hợp ma trận $X^T X$ là ma trận **khả nghịch** thì sẽ tồn tại duy nhất một nghiệm $W$.

$$
W = (X^T X)^{-1} (X^T Y)
$$

Còn trong trường hợp ma trận $X^T X$ có **định thức** bằng $0$ hay **không khả nghịch** thì ta có thể dùng ma trận **giả khả nghịch** của nó ký hiệu là $(X^T X)^{\dagger}$. Bạn có thể tự tìm tài liệu để đọc về nó, tại mình cũng không hiểu rõ về nó lắm. Nếu có thời gian thì mình sẽ tìm hiểu và viết một bài viết sau.

Tóm lại thì công thức tổng quát của $W$ sẽ là: 

$$
W = (X^T X)^{\dagger} (X^T Y)
$$

``` python
import numpy as np

x1_list = np.random.rand(500, 1) # tạo mảng x gồm các số ngẫu nhiên

y_list = 4 + 10 * x1_list + 0.2 * np.random.randn(x1_list.shape[0], 1)
#với w_1 = 10 và w_2 = 4, tạo ra các giá trị y tương ứng nhưng bị lệch đi một tí

ones = np.ones((x1_list.shape[0], 1))
X = np.concatenate((ones, x1_list), axis = 1)
# tạo ma trân X

W = np.dot(np.linalg.pinv(np.dot(X.T, X)), np.dot(X.T, y_list))
# Ma thuật của đại số tuyến tính
print(W)
```


Hoặc bạn có thể đọc qua [notebook này](https://github.com/orz-lab/orz-lab.github.io/blob/main/Notebooks/2024-10-13-Machine-Learning/Linear_Regression.ipynb) để có thể coi được thêm hình minh họa.

## Bằng Gradient Descent
Đây chính là thuật toán phổ biến, được dùng để train cho nhiều mô hình AI mà mình sẽ nói đến sau này.

### Ý tưởng của Gradient Descent
Ý tưởng của **Gradient Descent** thường được so sánh với việc **leo núi ngược** để dễ hình dung. Tuy nhiên, thay vì leo lên đỉnh, Gradient Descent tìm cách đi xuống để đạt điểm thấp nhất của hàm đó.

Khi thực hiện Gradient Descent, bạn luôn đi theo hướng dốc nhất đi xuống, nghĩa là tại mỗi điểm bạn sẽ tính toán **Gradient** (đạo hàm) để biết hướng mà hàm mất mát giảm mạnh nhất và di chuyển ngược lại với hướng của Gradient.

Ở đây ta có hàm $y = x^2$ và đạo hàm của nó là $y = 2x$.

![Ảnh 2]({{page.image}}/gradient_on_1d.gif)

Ta có thể thấy nếu điểm $I$ mà đang ở phía bên trái của **cực tiểu** thì đạo hàm sẽ **âm**, lúc này nếu ta đi ngược lại với đạo hàm tức là sẽ đi về phía **chiều dương**. Ngược lại nếu điểm $I$ đang ở bên phải thì đạo hàm sẽ **dương** và ta sẽ cần đi về phía **chiều âm**.

Ta sẽ thực hiện bức này cho đến khi nào đạo hàm xấp xỉ $0$ tức là giá trị đã chạm hoặc gần chạm cực tiểu.

### Gradient Descent
**Bài toán:** Tìm $x$ sao cho tại $x$ thì $f(x)$ đạt cực tiểu.

* Ta sẽ gán $x$ với một giá trị ngẫu nhiên nào đó.
* Lặp lại việc $x = x - \alpha \frac{\partial{f}}{\partial{x}}$ với $\alpha$ là một hằng số rất nhỏ, đến khi nào $\frac{\partial{f}}{\partial{x}}$ gần bằng không.

Ta có thể hình dung bài toán giống như việc ta **thả một viên bi** từ một vị trí bất kì trên đồ thị rồi đợi cho viên bị đó **lăn xuống** dưới vực và đến được vị trí thấp nhất có thể.

| ![Ảnh 3](https://miro.medium.com/v2/resize:fit:720/format:webp/1*HCrKH4lCWnPUfV7ZN_vtzg.jpeg) | 
|:--:| 
| *ảnh được trích từ ["GRADIENT DESCENT — A Journey, not a Destination to reach Global Minima"](https://datamantra.medium.com/gradient-descent-a-journey-not-a-destination-to-reach-global-minima-57793827a84b) của DataMantra, 2024* |

Quay trở lại với bài toán cũ ở phần đầu, lúc này ta không chỉ có một biến $x$ mà là 2 biến $a$ và $b$. Cũng tương tự với hàm 1 biến, ta sẽ làm như sau:

* Ta sẽ gán $a, b$ với một giá trị ngẫu nhiên nào đó.
* Lặp lại việc $a = a - \alpha \frac{\partial{L}}{\partial{a}}; b = b - \beta \frac{\partial{L}}{\partial{b}}$ với $\alpha$ và $\beta$ là một hằng số rất nhỏ.

***Ai đó:** Vậy tại sao việc ta tối ưu hàm $L$ với từng giá trị $a, b$ riêng biệt lại có thể tìm được cực tiểu của hàm số đó?*

Giải thích chi tiết thì sẽ hơi khó hiểu, nhưng cứ tưởng tượng bạn đang leo núi (lúc này đang ở không gian 3 chiều) thì việc mà bạn đi theo 2 hướng khác nhau và đều hướng xuống dưới vị trí thấp hơn vị trí bạn đang đứng thì đến một lúc nào đó thì bạn sẽ đên được cực tiểu.

| ![Ảnh 4](https://miro.medium.com/v2/resize:fit:640/format:webp/0*eHaQJx7HENmTZXjL.gif) | 
|:--:| 
| *ảnh được trích từ ["GRADIENT DESCENT — A Journey, not a Destination to reach Global Minima"](https://datamantra.medium.com/gradient-descent-a-journey-not-a-destination-to-reach-global-minima-57793827a84b) của DataMantra, 2024* |

# Bài toán với nhiều biến
Giờ thay vì là bài toán đưa ra **diện tích đất** và dự đoán **giá đất**, thì giờ ta sẽ đưa ra **diện tích đất**, **tỉ lệ đất có thể xây nhà** và ta phải dự đoán được **giá đất**.

Tương tự với bài toán ở đầu ta, ta đặt $m$ là **số lượng bộ dữ liệu**, $x_1$ là **diện tích đất**, $x_2$ là **tỉ lệ đất có thể xây nhà**, ta giả sử là có thể dự đoán bằng một phương trình tuyến tính có dạng như sau:

$$
\hat{y} = w_1 x_1 + w_2 x_2 + b
$$

Và lúc này hàm mất mát của chúng ta sẽ có dạng như sau:

$$
    L(w_1, w_2, b) = \sum_{i = 1}^{m}(\hat{y^{(i)}} - y^{(i)})^2 \\
    L(w_1, w_2, b) = \sum_{i = 1}^{m}(w_1 x_1^{(i)} + w_2 x_2^{(i)} + b - y^{(i)})^2 \\
$$

Ta cũng có thể dùng **Gradient Descent** để tìm được $w_1, w_2, b$ để hàm $L$ của chúng ta đạt giá trị nhỏ nhất.

* Ta sẽ gán $w_1, w_2, b$ với một giá trị ngẫu nhiên nào đó.
* Lặp lại việc sau:

$$
w_1 = w_1 - \beta \frac{\partial{L}}{\partial{w_1}} \\
w_2 = w_2 - \beta \frac{\partial{L}}{\partial{w_2}} \\
b = b - \beta \frac{\partial{L}}{\partial{b}} \\
$$

Tổng quát với trường hợp có $n$ biến đầu vào:

$$
    \hat{y} = w_1 x_1 + w_2 x_2 + w_3 x_3 + ... + + w_n x_n  + b \\
    L(w_1, w_2, w_3, ..., w_n, b) = \sum_{i = 1}^{m}(\hat{y^{(i)}} - y^{(i)})^2 \\
$$

Ở bước **Gradient Descent**:

$$
w_1 = w_1 - \beta \frac{\partial{L}}{\partial{w_1}} \\
w_2 = w_2 - \beta \frac{\partial{L}}{\partial{w_2}} \\
w_3 = w_3 - \beta \frac{\partial{L}}{\partial{w_3}} \\
... \\
w_n = w_n - \beta \frac{\partial{L}}{\partial{w_n}} \\
b = b - \beta \frac{\partial{L}}{\partial{b}} \\
$$

# Tài liệu tham khảo:
* DataMantra. (2024, June 17). GRADIENT DESCENT — A Journey, not a Destination to reach Global Minima. Medium. [https://datamantra.medium.com/gradient-descent-a-journey-not-a-destination-to-reach-global-minima-57793827a84b](https://datamantra.medium.com/gradient-descent-a-journey-not-a-destination-to-reach-global-minima-57793827a84b)

* Vu, T. (2016, December 28). Bài 3: Linear Regression. Tiep Vu’s Blog. [https://machinelearningcoban.com/2016/12/28/linearregression/](https://machinelearningcoban.com/2016/12/28/linearregression/)