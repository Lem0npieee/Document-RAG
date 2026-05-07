const fs = await import("node:fs/promises");
const path = await import("node:path");
const { Presentation, PresentationFile } = await import("@oai/artifact-tool");

const W = 1280;
const H = 720;

const DECK_ID = "document-rag-proposal-8page";
const OUT_DIR = "/Users/lianglihang/Downloads/Document-RAG";
const SCRATCH_DIR = path.resolve(process.env.PPTX_SCRATCH_DIR || path.join("tmp", "slides", DECK_ID));
const PREVIEW_DIR = path.join(SCRATCH_DIR, "preview");
const VERIFICATION_DIR = path.join(SCRATCH_DIR, "verification");
const INSPECT_PATH = path.join(SCRATCH_DIR, "inspect.ndjson");

// Images
const ARCH_IMAGE = "/tmp/pptx_extract/ppt/media/image1.png";
const DEMO_IMAGE = "/tmp/pptx_extract/ppt/media/image2.png";

const BG = "#F3F2EE";
const SURFACE = "#FFFEFC";
const SURFACE_ALT = "#F4F8F6";
const INK = "#142A2B";
const BODY = "#586D70";
const MUTE = "#7A8D90";
const ACCENT = "#0E7D6A";
const ACCENT_SOFT = "#E6F1EE";
const ACCENT_DEEP = "#0A5C50";
const BORDER = "#D7E1DE";
const WARM = "#F4EFE6";
const PEACH = "#EFE6D9";

const TITLE_FACE = "PingFang SC";
const BODY_FACE = "PingFang SC";
const MONO_FACE = "Aptos";

const SOURCES = {
  proposal: "Document-RAG/开题报告.md",
};

const SLIDES = [
  {
    kicker: "多模态大模型原理与应用 · 题目 3",
    title: "文档理解与多模态 GraphRAG 检索问答系统",
    subtitle: "基于 qwen3-vl-8b-instruct、FAISS 与 networkx 的可追溯文档智能助手",
    tags: ["多模态 GraphRAG", "DocVQA / ChartQA", "qwen3-vl-8b-instruct", "FAISS", "networkx"],
    footer: [
      ["核心技术栈", "qwen3-vl-8b-instruct + FAISS + networkx + DashScope"],
      ["当前状态", "原型系统已完成，进入评测阶段"],
    ],
    notes: "封面强调三件事：课程主线、核心技术栈、当前原型状态。",
    sources: ["proposal"],
  },
  {
    kicker: "01 · 任务背景",
    title: "为什么纯文本 RAG 不够",
    cards: [
      ["A", "视觉信息丢失", "图表趋势、版面结构、颜色标记、坐标轴和表格边界很难被纯文本准确保留。"],
      ["B", "结构关系不足", "同一页内的图文对应、跨页段落延续、图号与结论之间的支撑关系无法仅靠向量相似度稳定表达。"],
      ["C", "证据链不可解释", "向量检索可以找到相似片段，但不擅长回答\"某结论由哪些图表和实验支撑\"这类多跳问题。"],
    ],
    stripTitle: "系统要做什么",
    stripBody: "上传多页 PDF 或图像后，系统同时构建文本/图表片段、页面图像证据、关键词/关系节点和跨页连接，使回答既能利用多模态证据，又能给出可追溯的页码、节点和关系依据。",
    notes: "任务背景：讲清三个痛点和系统目标。",
    sources: ["proposal"],
  },
  {
    kicker: "02 · 系统总览",
    title: "系统怎么跑起来",
    subtitle: "知识库构建 + 在线问答 + 前端展示三条链路",
    notes: "架构图页：使用系统总览图作为主视觉。",
    sources: ["proposal"],
  },
  {
    kicker: "03 · 实验设计",
    title: "评测什么、怎么评",
    datasets: [
      ["数据集 A", "自建长文档测试集", "3-5 篇含图表、表格、跨页段落和结论引用的论文，人工编写问题。"],
      ["数据集 B", "DocVQA 子集", "公开 DocVQA 验证样本，评估单页文档问答抽取式准确性。"],
      ["数据集 C", "ChartQA 子集", "图表推理问题，观察页面原图和图表描述的作用。"],
    ],
    experiments: [
      ["消融实验", "Full / No-Graph / No-Image / Short-Answer", [
        "Full：向量召回 + 图谱扩展 + 社区 profile + 页面图像",
        "No-Graph：关闭图谱扩展，观察多跳问题下降幅度",
        "No-Image：关闭页面原图输入，观察图表题下降幅度",
        "Short-Answer Eval：约束输出为短答案，计算 ANLS/EM",
      ]],
    ],
    notes: "实验设计：展示数据集和消融矩阵。",
    sources: ["proposal"],
  },
  {
    kicker: "04 · Demo",
    title: "Knowledge Flow Studio",
    subtitle: "问答、知识图谱、节点检查、页面预览四块功能",
    notes: "Demo图页：使用当前系统截图。",
    sources: ["proposal"],
  },
  {
    kicker: "05 · 算力和计划",
    title: "算力预算与时间计划",
    resources: [
      ["本地计算", "CPU 即可", "PDF 渲染、FAISS、networkx 和 Flask 均可本地运行"],
      ["GPU", "非必需", "使用 DashScope API 调用 qwen3-vl-8b-instruct"],
      ["存储", "百 MB 到数 GB", "页面图片、FAISS 索引、JSON 解析结果和图谱文件"],
      ["API 成本", "与页数和问题数相关", "先小样本评测，再扩展；复用 build meta 避免重复入库"],
    ],
    progress: [
      ["工程骨架", "已完成"],
      ["文档解析", "已完成初版"],
      ["向量索引", "已完成"],
      ["图谱构建", "已完成初版"],
      ["GraphRAG 问答", "已完成初版"],
      ["Web 演示", "已完成初版"],
      ["评测脚本", "已完成框架"],
    ],
    plan: [
      ["第 1 周", "清理图谱噪声、优化关键词和关系过滤"],
      ["第 2 周", "构建自建长文档测试集，补充 DocVQA 小样本"],
      ["第 3 周", "跑 Full/No-Graph/No-Image 消融实验"],
      ["第 4 周", "完善前端演示、整理最终报告和答辩材料"],
    ],
    notes: "算力预算与时间计划：合并资源、进度和后续计划。",
    sources: ["proposal"],
  },
  {
    kicker: "06 · 参考文献",
    title: "核心来源",
    refs: [
      "[1] Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.",
      "[2] Edge, D. et al. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv:2404.16130.",
      "[3] Microsoft GraphRAG Documentation. Query Engine Overview and Indexing Overview.",
      "[4] Mathew, M. et al. (2021). DocVQA: A Dataset for VQA on Document Images. WACV 2021.",
      "[5] Masry, A. et al. (2022). ChartQA: A Benchmark for Question Answering about Charts. Findings of ACL 2022.",
      "[6] Faysse, M. et al. (2024). ColPali: Efficient Document Retrieval with Vision Language Models. arXiv:2407.01449.",
      "[7] Qwen Team. Qwen3-VL official repository and Qwen3-VL-8B-Instruct model card.",
      "[8] Dong, K. et al. (2025). Benchmarking Retrieval-Augmented Multimodal Generation for Document Question Answering. MMDocRAG.",
      "[9] Bu, C. et al. (2025). Query-Driven Multimodal GraphRAG: Dynamic Local Knowledge Graph Construction for Online Reasoning. Findings of ACL 2025.",
    ],
    notes: "参考文献：选 9 条核心来源。",
    sources: ["proposal"],
  },
  {
    kicker: "07 · 结束",
    title: "谢谢",
    subtitle: "多模态 GraphRAG · 可追溯文档问答 · qwen3-vl-8b-instruct + FAISS + networkx",
    closing: "从入库开始同时构建向量索引和文档图谱，使文本、图表、页面和跨页关系处于同一证据空间。",
    notes: "结束页：一句收束，不写模板句。",
    sources: ["proposal"],
  },
];

const inspectRecords = [];

function lineConfig(fill = "#00000000", width = 0) {
  return { style: "solid", fill, width };
}

function normalizeText(text) {
  if (Array.isArray(text)) {
    return text.map((item) => String(item ?? "")).join("\n");
  }
  return String(text ?? "");
}

function textLineCount(text) {
  const value = normalizeText(text);
  if (!value.trim()) return 0;
  return Math.max(1, value.split(/\n/).length);
}

function requiredTextHeight(text, fontSize, lineHeight = 1.22, minHeight = 8) {
  const lines = textLineCount(text);
  if (lines === 0) return minHeight;
  return Math.max(minHeight, lines * fontSize * lineHeight);
}

function assertTextFits(text, boxHeight, fontSize, role = "text") {
  const required = requiredTextHeight(text, fontSize);
  const tolerance = Math.max(2, fontSize * 0.12);
  if (normalizeText(text).trim() && boxHeight + tolerance < required) {
    throw new Error(
      `${role} text box too short: height=${boxHeight.toFixed(1)}, required=${required.toFixed(1)}, text=${JSON.stringify(
        normalizeText(text).slice(0, 80),
      )}`,
    );
  }
}

function recordShape(slideNo, shape, role, shapeType, x, y, w, h) {
  inspectRecords.push({
    kind: "shape",
    slide: slideNo,
    id: shape?.id || `shape-${slideNo}-${inspectRecords.length + 1}`,
    role,
    shapeType,
    bbox: [x, y, w, h],
  });
}

function recordText(slideNo, shape, role, text, x, y, w, h) {
  const value = normalizeText(text);
  inspectRecords.push({
    kind: "textbox",
    slide: slideNo,
    id: shape?.id || `text-${slideNo}-${inspectRecords.length + 1}`,
    role,
    text: value,
    textChars: value.length,
    textLines: textLineCount(value),
    bbox: [x, y, w, h],
  });
}

function addShape(slide, slideNo, geometry, x, y, w, h, fill = "#00000000", line = "#00000000", lineWidth = 0, role = geometry) {
  const shape = slide.shapes.add({
    geometry,
    position: { left: x, top: y, width: w, height: h },
    fill,
    line: lineConfig(line, lineWidth),
  });
  recordShape(slideNo, shape, role, geometry, x, y, w, h);
  return shape;
}

function addText(
  slide,
  slideNo,
  text,
  x,
  y,
  w,
  h,
  {
    size = 20,
    color = INK,
    bold = false,
    face = BODY_FACE,
    align = "left",
    valign = "top",
    fill = "#00000000",
    line = "#00000000",
    lineWidth = 0,
    role = "text",
    checkFit = true,
  } = {},
) {
  if (checkFit) {
    assertTextFits(text, h, size, role);
  }
  const box = addShape(slide, slideNo, "rect", x, y, w, h, fill, line, lineWidth, role);
  box.text = normalizeText(text);
  box.text.fontSize = size;
  box.text.color = color;
  box.text.bold = Boolean(bold);
  box.text.alignment = align;
  box.text.verticalAlignment = valign;
  box.text.typeface = face;
  box.text.insets = { left: 0, right: 0, top: 0, bottom: 0 };
  recordText(slideNo, box, role, text, x, y, w, h);
  return box;
}

function addPanel(slide, slideNo, x, y, w, h, { fill = SURFACE, line = BORDER, radiusRole = "panel", accent = null } = {}) {
  addShape(slide, slideNo, "roundRect", x, y, w, h, fill, line, 1.1, radiusRole);
  if (accent) {
    addShape(slide, slideNo, "rect", x, y, 6, h, accent, "#00000000", 0, `${radiusRole} accent`);
  }
}

function addBulletList(slide, slideNo, items, x, y, w, h, { size = 18, color = BODY, role = "list" } = {}) {
  const text = items.map((item) => `• ${item}`).join("\n");
  addText(slide, slideNo, text, x, y, w, h, {
    size,
    color,
    face: BODY_FACE,
    role,
  });
}

function addHeader(slide, slideNo, kicker) {
  addShape(slide, slideNo, "rect", 78, 57, 4, 18, ACCENT, "#00000000", 0, "header accent");
  addText(slide, slideNo, kicker, 92, 54, 460, 22, {
    size: 12,
    color: ACCENT_DEEP,
    bold: true,
    face: MONO_FACE,
    role: "header kicker",
    checkFit: false,
  });
  addShape(slide, slideNo, "roundRect", 1126, 46, 82, 34, SURFACE_ALT, BORDER, 1, "header index pill");
  addText(slide, slideNo, `${String(slideNo).padStart(2, "0")} / ${String(SLIDES.length).padStart(2, "0")}`, 1140, 56, 54, 14, {
    size: 12,
    color: ACCENT_DEEP,
    bold: true,
    face: MONO_FACE,
    align: "center",
    role: "header index",
    checkFit: false,
  });
  addShape(slide, slideNo, "rect", 78, 94, 1130, 1.5, BORDER, "#00000000", 0, "header line");
}

function addTitle(slide, slideNo, title, subtitle = null, x = 74, y = 92, w = 900) {
  addText(slide, slideNo, title, x, y, w, 112, {
    size: 40,
    color: INK,
    bold: true,
    face: TITLE_FACE,
    role: "title",
  });
  if (subtitle) {
    addText(slide, slideNo, subtitle, x, y + 112, Math.min(w, 820), 56, {
      size: 17,
      color: BODY,
      face: BODY_FACE,
      role: "subtitle",
    });
  }
}

function addBackground(slide, slideNo, variant = "default") {
  slide.background.fill = BG;
  addShape(slide, slideNo, "rect", 0, 0, W, 88, "#ECEFEB", "#00000000", 0, "bg band");
  addShape(slide, slideNo, "rect", 1030, 0, 250, 88, ACCENT_SOFT, "#00000000", 0, "bg accent block");
  addShape(slide, slideNo, "rect", 0, 614, 248, 106, variant === "warm" ? PEACH : "#EAF3F0", "#00000000", 0, "bg footer block");
  addShape(slide, slideNo, "roundRect", 40, 40, 1200, 640, SURFACE, BORDER, 1.05, "canvas");
  addShape(slide, slideNo, "rect", 40, 40, 10, 640, variant === "warm" ? WARM : SURFACE_ALT, "#00000000", 0, "canvas spine");
}

function addNotes(slide, body, sourceKeys) {
  const sourceLines = (sourceKeys || []).map((key) => `- ${SOURCES[key] || key}`).join("\n");
  slide.speakerNotes.setText(`${body}\n\n[Sources]\n${sourceLines}`);
}

// Slide 1: Cover
function slideCover(presentation) {
  const slideNo = 1;
  const data = SLIDES[0];
  const slide = presentation.slides.add();
  addBackground(slide, slideNo, "warm");

  addShape(slide, slideNo, "rect", 82, 118, 6, 438, ACCENT, "#00000000", 0, "cover rule");
  addText(slide, slideNo, "课程作业开题答辩", 106, 116, 220, 18, {
    size: 12,
    color: ACCENT_DEEP,
    bold: true,
    face: MONO_FACE,
    role: "cover kicker",
    checkFit: false,
  });
  addText(slide, slideNo, data.kicker, 106, 148, 360, 20, {
    size: 12,
    color: MUTE,
    bold: true,
    face: MONO_FACE,
    role: "cover course label",
    checkFit: false,
  });
  addText(slide, slideNo, data.title, 100, 188, 670, 136, {
    size: 48,
    color: INK,
    bold: true,
    face: TITLE_FACE,
    role: "cover title",
  });
  addText(slide, slideNo, data.subtitle, 106, 338, 608, 66, {
    size: 19,
    color: BODY,
    role: "cover subtitle",
  });

  addPanel(slide, slideNo, 826, 130, 340, 262, { fill: SURFACE_ALT, line: BORDER, radiusRole: "cover summary panel", accent: ACCENT });
  addText(slide, slideNo, "核心技术栈", 856, 156, 120, 18, {
    size: 13,
    color: ACCENT_DEEP,
    bold: true,
    face: MONO_FACE,
    role: "cover summary label",
    checkFit: false,
  });
  addBulletList(slide, slideNo, [
    "qwen3-vl-8b-instruct 文档解析",
    "FAISS 向量索引 + networkx 图谱",
    "DashScope API 多模态问答",
  ], 856, 194, 258, 92, {
    size: 18,
    color: BODY,
    role: "cover summary list",
  });
  addShape(slide, slideNo, "rect", 856, 310, 230, 2, "#BFDAD2", "#00000000", 0, "cover summary rule");
  addText(slide, slideNo, "原型已完成 · 进入评测阶段", 856, 332, 244, 34, {
    size: 16,
    color: INK,
    bold: true,
    role: "cover summary closing",
  });

  let tagX = 104;
  for (const tag of data.tags) {
    const width = Math.max(110, Math.min(184, tag.length * 15));
    addShape(slide, slideNo, "roundRect", tagX, 440, width, 32, SURFACE, BORDER, 1, "tag");
    addText(slide, slideNo, tag, tagX + 14, 448, width - 28, 16, {
      size: 12,
      color: ACCENT_DEEP,
      bold: true,
      face: BODY_FACE,
      role: "tag text",
      checkFit: false,
    });
    tagX += width + 12;
  }

  addPanel(slide, slideNo, 104, 518, 308, 90, { fill: SURFACE, line: BORDER, radiusRole: "cover footer card", accent: ACCENT });
  addPanel(slide, slideNo, 430, 518, 308, 90, { fill: SURFACE, line: BORDER, radiusRole: "cover footer card", accent: ACCENT });

  addText(slide, slideNo, data.footer[0][0], 128, 536, 120, 18, {
    size: 12,
    color: MUTE,
    bold: true,
    role: "cover footer label",
    checkFit: false,
  });
  addText(slide, slideNo, data.footer[0][1], 128, 560, 248, 34, {
    size: 16,
    color: INK,
    bold: true,
    role: "cover footer body",
  });
  addText(slide, slideNo, data.footer[1][0], 454, 536, 120, 18, {
    size: 12,
    color: MUTE,
    bold: true,
    role: "cover footer label",
    checkFit: false,
  });
  addText(slide, slideNo, data.footer[1][1], 454, 560, 248, 34, {
    size: 16,
    color: INK,
    bold: true,
    role: "cover footer body",
  });

  addShape(slide, slideNo, "rect", 826, 414, 340, 2, BORDER, "#00000000", 0, "cover lower rule");
  addText(slide, slideNo, "项目关键词", 856, 434, 80, 16, {
    size: 12,
    color: MUTE,
    bold: true,
    face: MONO_FACE,
    role: "cover lower label",
    checkFit: false,
  });
  addText(slide, slideNo, "多模态 GraphRAG · 可追溯文档问答", 856, 460, 270, 78, {
    size: 19,
    color: ACCENT_DEEP,
    bold: true,
    face: TITLE_FACE,
    role: "hero note",
  });
  addText(slide, slideNo, "向量检索 + 图谱扩展 + 页面图像融合", 856, 556, 264, 34, {
    size: 14,
    color: BODY,
    role: "hero note caption",
  });

  addNotes(slide, data.notes, data.sources);
}

// Slide 2: Task Background
function slideBackground(presentation) {
  const slideNo = 2;
  const data = SLIDES[1];
  const slide = presentation.slides.add();
  addBackground(slide, slideNo);
  addHeader(slide, slideNo, data.kicker);
  addTitle(slide, slideNo, data.title);

  const cardY = 254;
  const cardW = 350;
  const gap = 24;
  for (let i = 0; i < data.cards.length; i += 1) {
    const [index, title, body] = data.cards[i];
    const x = 76 + i * (cardW + gap);
    addPanel(slide, slideNo, x, cardY, cardW, 250, { fill: SURFACE, line: BORDER, radiusRole: "problem card", accent: ACCENT });
    addShape(slide, slideNo, "ellipse", x + 24, cardY + 22, 34, 34, ACCENT_SOFT, "#00000000", 0, "problem index");
    addText(slide, slideNo, index, x + 35, cardY + 31, 12, 12, {
      size: 12,
      color: ACCENT_DEEP,
      bold: true,
      role: "problem index text",
      checkFit: false,
    });
    addText(slide, slideNo, title, x + 24, cardY + 72, 288, 30, {
      size: 24,
      color: INK,
      bold: true,
      face: TITLE_FACE,
      role: "problem title",
    });
    addText(slide, slideNo, body, x + 24, cardY + 122, 298, 92, {
      size: 17,
      color: BODY,
      role: "problem body",
    });
  }

  addPanel(slide, slideNo, 76, 540, 1124, 104, { fill: SURFACE_ALT, line: BORDER, radiusRole: "focus strip" });
  addText(slide, slideNo, data.stripTitle, 108, 560, 120, 18, {
    size: 13,
    color: ACCENT_DEEP,
    bold: true,
    role: "focus label",
    checkFit: false,
  });
  addText(slide, slideNo, data.stripBody, 108, 590, 1040, 34, {
    size: 18,
    color: INK,
    bold: true,
    role: "focus body",
  });

  addNotes(slide, data.notes, data.sources);
}

// Slide 3: Architecture (with image)
function slideArchitecture(presentation, imageBytes) {
  const slideNo = 3;
  const data = SLIDES[2];
  const slide = presentation.slides.add();
  addBackground(slide, slideNo);
  addHeader(slide, slideNo, data.kicker);
  addTitle(slide, slideNo, data.title, data.subtitle, 74, 90, 900);

  // Add architecture image
  const imgW = 1100;
  const imgH = 420;
  const imgX = 90;
  const imgY = 210;
  
  const imageShape = slide.shapes.add({
    geometry: "rect",
    position: { left: imgX, top: imgY, width: imgW, height: imgH },
    fill: "#00000000",
    line: lineConfig(BORDER, 1),
  });
  
  // Try to add image
  if (imageBytes) {
    imageShape.fill = { type: "image", image: imageBytes };
    imageShape.line = lineConfig(BORDER, 1);
  } else {
    // Fallback: show placeholder text
    addText(slide, slideNo, "[系统架构图]", imgX, imgY, imgW, imgH, {
      size: 24,
      color: MUTE,
      align: "center",
      valign: "middle",
      role: "image-placeholder",
    });
  }
  
  recordShape(slideNo, imageShape, "architecture-image", "rect", imgX, imgY, imgW, imgH);

  // Bottom caption
  addPanel(slide, slideNo, 90, 650, 1100, 50, { fill: SURFACE_ALT, line: BORDER, radiusRole: "arch-caption" });
  addText(slide, slideNo, "PyMuPDF → qwen3-vl-8b-instruct → FAISS + networkx → 多模态问答", 110, 662, 1060, 26, {
    size: 15,
    color: BODY,
    align: "center",
    role: "arch-caption-text",
  });

  addNotes(slide, data.notes, data.sources);
}

// Slide 4: Experiment Design
function slideExperiment(presentation) {
  const slideNo = 4;
  const data = SLIDES[3];
  const slide = presentation.slides.add();
  addBackground(slide, slideNo);
  addHeader(slide, slideNo, data.kicker);
  addTitle(slide, slideNo, data.title);

  const datasetY = 246;
  const datasetW = 350;
  for (let i = 0; i < data.datasets.length; i += 1) {
    const [tag, title, body] = data.datasets[i];
    const x = 76 + i * (datasetW + 22);
    addPanel(slide, slideNo, x, datasetY, datasetW, 146, { fill: SURFACE, line: BORDER, radiusRole: "dataset card" });
    addText(slide, slideNo, tag, x + 24, datasetY + 22, 86, 16, {
      size: 12,
      color: ACCENT_DEEP,
      bold: true,
      role: "dataset tag",
      checkFit: false,
    });
    addText(slide, slideNo, title, x + 24, datasetY + 48, 230, 24, {
      size: 24,
      color: INK,
      bold: true,
      face: TITLE_FACE,
      role: "dataset title",
      checkFit: false,
    });
    addText(slide, slideNo, body, x + 24, datasetY + 86, 292, 42, {
      size: 15,
      color: BODY,
      role: "dataset body",
    });
  }

  const experimentY = 430;
  for (let i = 0; i < data.experiments.length; i += 1) {
    const [label, title, items] = data.experiments[i];
    const x = 76 + i * 562;
    addPanel(slide, slideNo, x, experimentY, 542, 210, {
      fill: i === 1 ? SURFACE_ALT : SURFACE,
      line: BORDER,
      radiusRole: "experiment panel",
      accent: ACCENT,
    });
    addText(slide, slideNo, label, x + 28, experimentY + 24, 120, 16, {
      size: 12,
      color: ACCENT_DEEP,
      bold: true,
      role: "experiment label",
      checkFit: false,
    });
    addText(slide, slideNo, title, x + 28, experimentY + 50, 340, 52, {
      size: 21,
      color: INK,
      bold: true,
      face: TITLE_FACE,
      role: "experiment title",
      checkFit: false,
    });
    addBulletList(slide, slideNo, items, x + 28, experimentY + 116, 470, 100, {
      size: 15,
      color: BODY,
      role: "experiment list",
    });
  }

  addNotes(slide, data.notes, data.sources);
}

// Slide 5: Demo (with image)
function slideDemo(presentation, imageBytes) {
  const slideNo = 5;
  const data = SLIDES[4];
  const slide = presentation.slides.add();
  addBackground(slide, slideNo);
  addHeader(slide, slideNo, data.kicker);
  addTitle(slide, slideNo, data.title, data.subtitle, 74, 90, 900);

  // Add demo image
  const imgW = 1100;
  const imgH = 440;
  const imgX = 90;
  const imgY = 200;
  
  const imageShape = slide.shapes.add({
    geometry: "rect",
    position: { left: imgX, top: imgY, width: imgW, height: imgH },
    fill: "#00000000",
    line: lineConfig(BORDER, 1),
  });
  
  if (imageBytes) {
    imageShape.fill = { type: "image", image: imageBytes };
    imageShape.line = lineConfig(BORDER, 1);
  } else {
    addText(slide, slideNo, "[Demo 截图]", imgX, imgY, imgW, imgH, {
      size: 24,
      color: MUTE,
      align: "center",
      valign: "middle",
      role: "image-placeholder",
    });
  }
  
  recordShape(slideNo, imageShape, "demo-image", "rect", imgX, imgY, imgW, imgH);

  // Feature labels below image
  const features = ["问答", "知识图谱", "节点检查", "页面预览"];
  const featW = 240;
  const featGap = 50;
  const startX = 90 + (1100 - (features.length * featW + (features.length - 1) * featGap)) / 2;
  
  for (let i = 0; i < features.length; i++) {
    const fx = startX + i * (featW + featGap);
    addShape(slide, slideNo, "roundRect", fx, 660, featW, 28, ACCENT_SOFT, BORDER, 1, "feature-tag");
    addText(slide, slideNo, features[i], fx, 664, featW, 20, {
      size: 13,
      color: ACCENT_DEEP,
      bold: true,
      align: "center",
      role: "feature-tag-text",
      checkFit: false,
    });
  }

  addNotes(slide, data.notes, data.sources);
}

// Slide 6: Budget & Plan
function slideBudget(presentation) {
  const slideNo = 6;
  const data = SLIDES[5];
  const slide = presentation.slides.add();
  addBackground(slide, slideNo, "warm");
  addHeader(slide, slideNo, data.kicker);
  addTitle(slide, slideNo, data.title);

  // Left: Resources
  addPanel(slide, slideNo, 76, 246, 540, 380, { fill: SURFACE, line: BORDER, radiusRole: "resource panel", accent: ACCENT });
  addText(slide, slideNo, "算力与成本", 104, 272, 140, 20, {
    size: 15,
    color: ACCENT_DEEP,
    bold: true,
    role: "resource label",
    checkFit: false,
  });
  
  let ry = 310;
  for (const [name, value, detail] of data.resources) {
    addText(slide, slideNo, name, 104, ry, 100, 18, {
      size: 14,
      color: ACCENT_DEEP,
      bold: true,
      role: "resource name",
      checkFit: false,
    });
    addText(slide, slideNo, value, 210, ry, 100, 18, {
      size: 14,
      color: INK,
      bold: true,
      role: "resource value",
      checkFit: false,
    });
    addText(slide, slideNo, detail, 104, ry + 22, 480, 30, {
      size: 13,
      color: BODY,
      role: "resource detail",
    });
    ry += 62;
  }

  // Right: Progress & Plan
  addPanel(slide, slideNo, 640, 246, 560, 180, { fill: SURFACE_ALT, line: BORDER, radiusRole: "progress panel", accent: ACCENT });
  addText(slide, slideNo, "当前进度", 668, 272, 120, 20, {
    size: 15,
    color: ACCENT_DEEP,
    bold: true,
    role: "progress label",
    checkFit: false,
  });
  
  const progressText = data.progress.map(([item, status]) => `• ${item}：${status}`).join("\n");
  addText(slide, slideNo, progressText, 668, 304, 504, 110, {
    size: 13,
    color: BODY,
    role: "progress list",
  });

  addPanel(slide, slideNo, 640, 446, 560, 180, { fill: SURFACE, line: BORDER, radiusRole: "plan panel", accent: ACCENT });
  addText(slide, slideNo, "后续 4 周计划", 668, 472, 140, 20, {
    size: 15,
    color: ACCENT_DEEP,
    bold: true,
    role: "plan label",
    checkFit: false,
  });
  
  const planText = data.plan.map(([week, work]) => `• ${week}：${work}`).join("\n");
  addText(slide, slideNo, planText, 668, 504, 504, 110, {
    size: 13,
    color: BODY,
    role: "plan list",
  });

  addNotes(slide, data.notes, data.sources);
}

// Slide 7: References
function slideReferences(presentation) {
  const slideNo = 7;
  const data = SLIDES[6];
  const slide = presentation.slides.add();
  addBackground(slide, slideNo);
  addHeader(slide, slideNo, data.kicker);
  addTitle(slide, slideNo, data.title);

  addPanel(slide, slideNo, 76, 246, 1124, 380, { fill: SURFACE, line: BORDER, radiusRole: "refs panel", accent: ACCENT });
  
  const refsText = data.refs.join("\n\n");
  addText(slide, slideNo, refsText, 104, 272, 1068, 340, {
    size: 14,
    color: BODY,
    role: "refs list",
  });

  addNotes(slide, data.notes, data.sources);
}

// Slide 8: Closing
function slideClosing(presentation) {
  const slideNo = 8;
  const data = SLIDES[7];
  const slide = presentation.slides.add();
  addBackground(slide, slideNo, "warm");

  addShape(slide, slideNo, "rect", 82, 118, 6, 438, ACCENT, "#00000000", 0, "closing rule");
  addText(slide, slideNo, data.kicker, 106, 148, 360, 20, {
    size: 12,
    color: MUTE,
    bold: true,
    face: MONO_FACE,
    role: "closing kicker",
    checkFit: false,
  });
  addText(slide, slideNo, data.title, 100, 188, 600, 80, {
    size: 56,
    color: INK,
    bold: true,
    face: TITLE_FACE,
    role: "closing title",
  });
  addText(slide, slideNo, data.subtitle, 106, 290, 700, 50, {
    size: 19,
    color: BODY,
    role: "closing subtitle",
  });

  addPanel(slide, slideNo, 76, 380, 1124, 120, { fill: SURFACE_ALT, line: BORDER, radiusRole: "closing panel" });
  addText(slide, slideNo, data.closing, 104, 404, 1068, 72, {
    size: 18,
    color: INK,
    bold: true,
    role: "closing body",
  });

  // Keywords at bottom
  const keywords = ["GraphRAG", "FAISS", "networkx", "qwen3-vl-8b-instruct", "DashScope", "PyMuPDF", "Flask"];
  let kwX = 104;
  for (const kw of keywords) {
    const width = Math.max(100, kw.length * 14);
    addShape(slide, slideNo, "roundRect", kwX, 560, width, 32, SURFACE, BORDER, 1, "keyword-tag");
    addText(slide, slideNo, kw, kwX + 10, 566, width - 20, 20, {
      size: 12,
      color: ACCENT_DEEP,
      bold: true,
      align: "center",
      role: "keyword-text",
      checkFit: false,
    });
    kwX += width + 14;
  }

  addNotes(slide, data.notes, data.sources);
}

async function createDeck() {
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });
  
  // Read images
  let archImageBytes = null;
  let demoImageBytes = null;
  try {
    archImageBytes = await fs.readFile(ARCH_IMAGE);
  } catch (e) {
    console.warn("Could not read architecture image:", e.message);
  }
  try {
    demoImageBytes = await fs.readFile(DEMO_IMAGE);
  } catch (e) {
    console.warn("Could not read demo image:", e.message);
  }
  
  slideCover(presentation);
  slideBackground(presentation);
  slideArchitecture(presentation, archImageBytes);
  slideExperiment(presentation);
  slideDemo(presentation, demoImageBytes);
  slideBudget(presentation);
  slideReferences(presentation);
  slideClosing(presentation);
  return presentation;
}

async function ensureDirs() {
  await fs.mkdir(OUT_DIR, { recursive: true });
  await fs.mkdir(SCRATCH_DIR, { recursive: true });
  await fs.mkdir(PREVIEW_DIR, { recursive: true });
  await fs.mkdir(VERIFICATION_DIR, { recursive: true });
}

async function saveBlobToFile(blob, filePath) {
  const bytes = new Uint8Array(await blob.arrayBuffer());
  await fs.writeFile(filePath, bytes);
}

async function writeInspectArtifact(presentation) {
  const deckRecord = {
    kind: "deck",
    id: DECK_ID,
    slideCount: presentation.slides.count,
    slideSize: { width: W, height: H },
  };
  const slideRecords = presentation.slides.items.map((slide, index) => ({
    kind: "slide",
    slide: index + 1,
    id: slide?.id || `slide-${index + 1}`,
  }));
  const lines = [deckRecord, ...slideRecords, ...inspectRecords].map((record) => JSON.stringify(record)).join("\n") + "\n";
  await fs.writeFile(INSPECT_PATH, lines, "utf8");
}

async function verifyAndExport(presentation) {
  await ensureDirs();
  await writeInspectArtifact(presentation);
  for (let idx = 0; idx < presentation.slides.items.length; idx += 1) {
    const preview = await presentation.export({ slide: presentation.slides.items[idx], format: "png", scale: 1 });
    await saveBlobToFile(preview, path.join(PREVIEW_DIR, `slide-${String(idx + 1).padStart(2, "0")}.png`));
  }
  const pptxBlob = await PresentationFile.exportPptx(presentation);
  const pptxPath = path.join(OUT_DIR, "文档理解与多模态GraphRAG检索问答系统-开题展示.pptx");
  await pptxBlob.save(pptxPath);
  return pptxPath;
}

const presentation = await createDeck();
const pptxPath = await verifyAndExport(presentation);
console.log(pptxPath);
