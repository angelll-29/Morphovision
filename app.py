import streamlit as st
import cv2
import numpy as np
from PIL import Image
from io import BytesIO


# ---------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------

st.set_page_config(
    page_title="MorphoVision",
    page_icon="🧬",
    layout="wide"
)


# ---------------------------------------------------------
# CUSTOM CSS
# ---------------------------------------------------------

st.markdown("""
<style>
    .main-title {
        font-size: 48px;
        font-weight: 800;
        margin-bottom: 0px;
    }

    .subtitle {
        font-size: 19px;
        color: #9ca3af;
        margin-bottom: 25px;
    }

    .section-title {
        font-size: 28px;
        font-weight: 750;
        margin-top: 30px;
        margin-bottom: 5px;
    }

    .section-description {
        color: #9ca3af;
        margin-bottom: 20px;
    }

    .info-box {
        padding: 18px;
        border-radius: 12px;
        background: #172f49;
        border-left: 5px solid #60a5fa;
        margin-top: 15px;
        margin-bottom: 20px;
    }

    .operation-card {
        padding: 18px;
        border-radius: 12px;
        background: #252735;
        min-height: 150px;
        border: 1px solid #3b3f52;
    }

    .operation-card h4 {
        margin-top: 0px;
    }

    .challenge-box {
        padding: 22px;
        border-radius: 14px;
        background: #202c3d;
        border: 1px solid #3b82f6;
        margin-top: 15px;
    }

    .small-text {
        color: #a1a1aa;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------
# HELPER FUNCTIONS
# ---------------------------------------------------------

def convert_to_rgb(image):
    """Convert uploaded image into RGB format."""
    return np.array(image.convert("RGB"))


def create_binary_image(rgb_image):
    """Convert RGB image into grayscale and binary image."""
    gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY)

    _, binary = cv2.threshold(
        gray,
        0,
        255,
        cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    return gray, binary


def apply_morphology(binary, operation, kernel_size, iterations):
    """Apply selected morphological operation."""

    kernel = np.ones(
        (kernel_size, kernel_size),
        np.uint8
    )

    if operation == "None":
        result = binary.copy()

    elif operation == "Erosion":
        result = cv2.erode(
            binary,
            kernel,
            iterations=iterations
        )

    elif operation == "Dilation":
        result = cv2.dilate(
            binary,
            kernel,
            iterations=iterations
        )

    elif operation == "Opening":
        result = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            kernel,
            iterations=iterations
        )

    elif operation == "Closing":
        result = cv2.morphologyEx(
            binary,
            cv2.MORPH_CLOSE,
            kernel,
            iterations=iterations
        )

    return result


def find_largest_contour(binary):
    """Find the largest meaningful object contour."""

    contours, _ = cv2.findContours(
        binary,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE
    )

    valid_contours = [
        contour
        for contour in contours
        if cv2.contourArea(contour) > 100
    ]

    if not valid_contours:
        return None

    return max(valid_contours, key=cv2.contourArea)


def classify_shape(contour, circularity, aspect_ratio):
    """Classify the detected object more reliably."""

    perimeter = cv2.arcLength(contour, True)

    if perimeter == 0:
        return "Unknown Shape", 0

    # Use a smaller approximation factor for complex objects
    approximation = cv2.approxPolyDP(
        contour,
        0.02 * perimeter,
        True
    )

    vertices = len(approximation)

    area = cv2.contourArea(contour)

    # Convex hull helps identify complex or irregular objects
    hull = cv2.convexHull(contour)
    hull_area = cv2.contourArea(hull)

    solidity = (
        area / hull_area
        if hull_area != 0
        else 0
    )

    # Very elongated objects are usually not circles
    if circularity > 0.82 and 0.85 <= aspect_ratio <= 1.15:
        shape = "Circle"

    elif vertices == 3 and solidity > 0.85:
        shape = "Triangle"

    elif vertices == 4 and solidity > 0.85:
        if 0.90 <= aspect_ratio <= 1.10:
            shape = "Square"
        else:
            shape = "Rectangle"

    elif solidity < 0.75 or vertices > 8:
        shape = "Irregular Shape"

    else:
        shape = "Complex Shape"

    return shape, vertices


def measure_shape(binary):
    """Calculate geometric properties of the largest object."""

    contour = find_largest_contour(binary)

    if contour is None:
        return None

    area = cv2.contourArea(contour)
    perimeter = cv2.arcLength(contour, True)

    x, y, width, height = cv2.boundingRect(contour)

    aspect_ratio = width / height if height != 0 else 0

    circularity = (
        (4 * np.pi * area) / (perimeter * perimeter)
        if perimeter != 0
        else 0
    )

    moments = cv2.moments(contour)

    if moments["m00"] != 0:
        centroid_x = int(moments["m10"] / moments["m00"])
        centroid_y = int(moments["m01"] / moments["m00"])
    else:
        centroid_x, centroid_y = 0, 0

    shape, vertices = classify_shape(
        contour,
        circularity,
        aspect_ratio
    )

    return {
        "contour": contour,
        "area": area,
        "perimeter": perimeter,
        "width": width,
        "height": height,
        "aspect_ratio": aspect_ratio,
        "circularity": circularity,
        "centroid_x": centroid_x,
        "centroid_y": centroid_y,
        "vertices": vertices,
        "shape": shape
    }


def draw_analysis(rgb_image, measurements):
    """Draw contour, bounding box and centroid."""

    output = rgb_image.copy()

    if measurements is None:
        return output

    contour = measurements["contour"]

    x, y, width, height = cv2.boundingRect(contour)

    centroid_x = measurements["centroid_x"]
    centroid_y = measurements["centroid_y"]

    cv2.drawContours(
        output,
        [contour],
        -1,
        (0, 255, 0),
        3
    )

    cv2.rectangle(
        output,
        (x, y),
        (x + width, y + height),
        (0, 100, 255),
        2
    )

    cv2.circle(
        output,
        (centroid_x, centroid_y),
        6,
        (255, 0, 0),
        -1
    )

    cv2.putText(
        output,
        measurements["shape"],
        (x, max(y - 10, 25)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 120, 255),
        2
    )

    return output


def percentage_change(original, new):
    """Calculate percentage change."""

    if original == 0:
        return 0

    return ((new - original) / original) * 100


def explain_operation(operation):
    """Return a simple explanation for each operation."""

    explanations = {
        "None": "The original binary object is displayed without morphological changes.",

        "Erosion": (
            "Erosion removes pixels from the boundaries of foreground objects. "
            "It makes objects thinner, smaller and can remove tiny noise."
        ),

        "Dilation": (
            "Dilation adds pixels around foreground boundaries. "
            "It makes objects thicker, larger and can connect nearby regions."
        ),

        "Opening": (
            "Opening applies erosion followed by dilation. "
            "It is useful for removing small noise while preserving the main object."
        ),

        "Closing": (
            "Closing applies dilation followed by erosion. "
            "It is useful for filling small gaps and holes in an object."
        )
    }

    return explanations[operation]


def image_to_bytes(image_array):
    """Convert NumPy image to downloadable PNG bytes."""

    image = Image.fromarray(image_array)

    buffer = BytesIO()
    image.save(buffer, format="PNG")

    return buffer.getvalue()


# ---------------------------------------------------------
# SIDEBAR
# ---------------------------------------------------------

st.sidebar.markdown("## 🧬 Morphology Studio")
st.sidebar.write(
    "Experiment with image morphology and observe how object shapes change."
)

operation = st.sidebar.selectbox(
    "Choose Operation",
    [
        "None",
        "Erosion",
        "Dilation",
        "Opening",
        "Closing"
    ]
)

kernel_size = st.sidebar.slider(
    "Kernel Size",
    min_value=3,
    max_value=15,
    value=5,
    step=2
)

iterations = st.sidebar.slider(
    "Iterations",
    min_value=1,
    max_value=5,
    value=1
)

st.sidebar.markdown("---")

st.sidebar.markdown("""
### 📘 Operation Guide

**Erosion:** Shrinks objects  
**Dilation:** Expands objects  
**Opening:** Removes small noise  
**Closing:** Fills small gaps
""")

st.sidebar.markdown("---")

st.sidebar.info(
    "Tip: Use a simple black object on a plain white background for better results."
)


# ---------------------------------------------------------
# MAIN HEADER
# ---------------------------------------------------------

st.markdown(
    '<div class="main-title">🧬 MorphoVision</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="subtitle">See Shapes Shrink. Watch Them Grow.</div>',
    unsafe_allow_html=True
)

st.write(
    "An interactive Object Shape Analysis System using "
    "erosion, dilation and morphological image processing."
)

st.markdown("---")


# ---------------------------------------------------------
# IMAGE UPLOAD
# ---------------------------------------------------------

uploaded_file = st.file_uploader(
    "📤 Upload an image of an object",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is None:
    st.info(
        "Upload an image to begin the Morphology Studio experiment."
    )
    st.stop()


image = Image.open(uploaded_file)
rgb_image = convert_to_rgb(image)

gray_image, binary_image = create_binary_image(rgb_image)

processed_image = apply_morphology(
    binary_image,
    operation,
    kernel_size,
    iterations
)


# ---------------------------------------------------------
# MORPHOLOGY SIMULATOR
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🔬 Morphology Simulator</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Observe how different morphological operations transform the same object.'
    '</div>',
    unsafe_allow_html=True
)

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.image(
        rgb_image,
        caption="Original Image",
        use_container_width=True
    )

with col2:
    st.image(
        binary_image,
        caption="Binary Image",
        use_container_width=True
    )

with col3:
    erosion_image = apply_morphology(
        binary_image,
        "Erosion",
        kernel_size,
        iterations
    )

    st.image(
        erosion_image,
        caption="Erosion: Shrinks Object",
        use_container_width=True
    )

with col4:
    dilation_image = apply_morphology(
        binary_image,
        "Dilation",
        kernel_size,
        iterations
    )

    st.image(
        dilation_image,
        caption="Dilation: Expands Object",
        use_container_width=True
    )


st.markdown(
    f"""
    <div class="info-box">
    <b>Current Operation: {operation}</b><br>
    {explain_operation(operation)}
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# ADDITIONAL MORPHOLOGY COMPARISON
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🧪 Compare All Operations</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Compare the effect of erosion, dilation, opening and closing.'
    '</div>',
    unsafe_allow_html=True
)

open_col, close_col = st.columns(2)

with open_col:
    opening_image = apply_morphology(
        binary_image,
        "Opening",
        kernel_size,
        iterations
    )

    st.image(
        opening_image,
        caption="Opening: Removes Small Noise",
        use_container_width=True
    )

with close_col:
    closing_image = apply_morphology(
        binary_image,
        "Closing",
        kernel_size,
        iterations
    )

    st.image(
        closing_image,
        caption="Closing: Fills Small Gaps",
        use_container_width=True
    )


# ---------------------------------------------------------
# SHAPE ANALYSIS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📊 Geometric Shape Analysis</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Measure the geometric properties of the selected object.'
    '</div>',
    unsafe_allow_html=True
)

original_measurements = measure_shape(binary_image)
processed_measurements = measure_shape(processed_image)

if processed_measurements is None:
    st.warning(
        "No clear object was detected. Try using a simple object on a plain background."
    )
    st.stop()


analyzed_image = draw_analysis(
    rgb_image,
    processed_measurements
)

st.image(
    analyzed_image,
    caption="Detected Object with Contour, Bounding Box and Centroid",
    use_container_width=True
)

m1, m2, m3, m4 = st.columns(4)

with m1:
    st.metric(
        "Shape",
        processed_measurements["shape"]
    )

with m2:
    st.metric(
        "Area",
        f'{processed_measurements["area"]:.2f} px²'
    )

with m3:
    st.metric(
        "Perimeter",
        f'{processed_measurements["perimeter"]:.2f} px'
    )

with m4:
    st.metric(
        "Vertices",
        processed_measurements["vertices"]
    )

m5, m6, m7, m8 = st.columns(4)

with m5:
    st.metric(
        "Width",
        f'{processed_measurements["width"]} px'
    )

with m6:
    st.metric(
        "Height",
        f'{processed_measurements["height"]} px'
    )

with m7:
    st.metric(
        "Circularity",
        f'{processed_measurements["circularity"]:.3f}'
    )

with m8:
    st.metric(
        "Aspect Ratio",
        f'{processed_measurements["aspect_ratio"]:.3f}'
    )


# ---------------------------------------------------------
# SHAPE INSIGHT
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🕵️ Shape Detective</div>',
    unsafe_allow_html=True
)

shape_name = processed_measurements["shape"]
vertices = processed_measurements["vertices"]
circularity = processed_measurements["circularity"]
aspect_ratio = processed_measurements["aspect_ratio"]

st.markdown(
    f"""
    <div class="info-box">
    <b>Detected Shape: {shape_name}</b><br><br>
    The object has approximately <b>{vertices} vertices</b>,
    a circularity of <b>{circularity:.3f}</b>,
    and an aspect ratio of <b>{aspect_ratio:.3f}</b>.
    </div>
    """,
    unsafe_allow_html=True
)


# ---------------------------------------------------------
# SHAPE DOCTOR REPORT
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🩺 Morphology Impact Report</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Understand how morphology changes the object’s geometric condition.'
    '</div>',
    unsafe_allow_html=True
)

if original_measurements is not None:

    original_area = original_measurements["area"]
    erosion_measurements = measure_shape(erosion_image)
    dilation_measurements = measure_shape(dilation_image)

    report_col1, report_col2 = st.columns(2)

    with report_col1:
        st.markdown("### 🔻 Erosion Report")

        if erosion_measurements is not None:
            erosion_area = erosion_measurements["area"]
            erosion_change = percentage_change(
                original_area,
                erosion_area
            )

            st.metric(
                "Area Change",
                f"{erosion_change:.2f}%"
            )

            st.write(
                f"Erosion changed the area from "
                f"{original_area:.2f} px² to "
                f"{erosion_area:.2f} px²."
            )

            st.info(
                "Erosion removes boundary pixels and makes the object thinner or smaller."
            )

    with report_col2:
        st.markdown("### 🔺 Dilation Report")

        if dilation_measurements is not None:
            dilation_area = dilation_measurements["area"]
            dilation_change = percentage_change(
                original_area,
                dilation_area
            )

            st.metric(
                "Area Change",
                f"{dilation_change:.2f}%"
            )

            st.write(
                f"Dilation changed the area from "
                f"{original_area:.2f} px² to "
                f"{dilation_area:.2f} px²."
            )

            st.info(
                "Dilation adds boundary pixels and makes the object thicker or larger."
            )


# ---------------------------------------------------------
# TRANSFORMATION METER
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">📈 Shape Transformation Meter</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'This meter shows the amount of change caused by erosion and dilation.'
    '</div>',
    unsafe_allow_html=True
)

if original_measurements is not None:

    original_area = original_measurements["area"]

    erosion_area = (
        erosion_measurements["area"]
        if erosion_measurements is not None
        else 0
    )

    dilation_area = (
        dilation_measurements["area"]
        if dilation_measurements is not None
        else 0
    )

    erosion_change = percentage_change(
        original_area,
        erosion_area
    )

    dilation_change = percentage_change(
        original_area,
        dilation_area
    )

    meter1, meter2 = st.columns(2)

    with meter1:
        st.markdown("### 🔻 Erosion Effect")
        st.progress(
            min(abs(erosion_change) / 100, 1.0)
        )
        st.write(
            f"Area change after erosion: **{erosion_change:.2f}%**"
        )

    with meter2:
        st.markdown("### 🔺 Dilation Effect")
        st.progress(
            min(abs(dilation_change) / 100, 1.0)
        )
        st.write(
            f"Area change after dilation: **{dilation_change:.2f}%**"
        )


# ---------------------------------------------------------
# MORPHOLOGY CHALLENGE
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🎯 Morphology Challenge</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Test your understanding of morphological image processing.'
    '</div>',
    unsafe_allow_html=True
)

challenge = st.selectbox(
    "Which operation is best for removing small unwanted noise?",
    [
        "Choose an answer",
        "Erosion",
        "Dilation",
        "Opening",
        "Closing"
    ]
)

if challenge != "Choose an answer":

    if challenge == "Opening":
        st.success(
            "Correct! Opening removes small noise using erosion followed by dilation."
        )
    else:
        st.error(
            "Not quite. The correct answer is Opening."
        )

st.markdown("### More Quick Questions")

question1 = st.radio(
    "Which operation makes an object thicker?",
    [
        "Erosion",
        "Dilation",
        "Opening",
        "Closing"
    ],
    key="question1"
)

if question1 == "Dilation":
    st.success("Correct! Dilation expands foreground objects.")
else:
    st.info("Answer: Dilation")


question2 = st.radio(
    "Which operation fills small gaps in an object?",
    [
        "Erosion",
        "Dilation",
        "Opening",
        "Closing"
    ],
    key="question2"
)

if question2 == "Closing":
    st.success("Correct! Closing fills small gaps and holes.")
else:
    st.info("Answer: Closing")


# ---------------------------------------------------------
# REAL-WORLD APPLICATIONS
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">🌍 Real-World Applications</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-description">'
    'Morphological image processing is used in many practical fields.'
    '</div>',
    unsafe_allow_html=True
)

app1, app2, app3, app4 = st.columns(4)

with app1:
    st.markdown("""
    <div class="operation-card">
    <h4>🏥 Medical Imaging</h4>
    <p>Helps clean images and analyze the boundaries of cells and medical objects.</p>
    </div>
    """, unsafe_allow_html=True)

with app2:
    st.markdown("""
    <div class="operation-card">
    <h4>🏭 Industrial Inspection</h4>
    <p>Used to detect cracks, holes, defects and missing parts in products.</p>
    </div>
    """, unsafe_allow_html=True)

with app3:
    st.markdown("""
    <div class="operation-card">
    <h4>📄 Document Processing</h4>
    <p>Helps remove noise and connect broken characters in scanned documents.</p>
    </div>
    """, unsafe_allow_html=True)

with app4:
    st.markdown("""
    <div class="operation-card">
    <h4>🚗 Traffic Analysis</h4>
    <p>Helps improve object boundaries in vehicle and road-image detection.</p>
    </div>
    """, unsafe_allow_html=True)


# ---------------------------------------------------------
# DOWNLOAD SECTION
# ---------------------------------------------------------

st.markdown(
    '<div class="section-title">💾 Save Your Analysis</div>',
    unsafe_allow_html=True
)

download_data = image_to_bytes(analyzed_image)

st.download_button(
    label="📥 Download Analyzed Image",
    data=download_data,
    file_name="morphovision_analysis.png",
    mime="image/png"
)

st.markdown("---")

st.caption(
    "MorphoVision | Object Shape Analysis using Mathematical Morphology"
)