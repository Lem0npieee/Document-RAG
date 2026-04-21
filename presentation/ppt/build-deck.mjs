const fs = await import("node:fs/promises");
const path = await import("node:path");
const { Presentation, PresentationFile } = await import("@oai/artifact-tool");

const W = 1280;
const H = 720;

const DECK_ID = "document-rag-proposal";
const OUT_DIR = "/Users/lianglihang/Downloads/Document-RAG/presentation/ppt/outputs";
const SCRATCH_DIR = path.resolve(process.env.PPTX_SCRATCH_DIR || path.join("tmp", "slides", DECK_ID));
const PREVIEW_DIR = path.join(SCRATCH_DIR, "preview");
const VERIFICATION_DIR = path.join(SCRATCH_DIR, "verification");
const INSPECT_PATH = path.join(SCRATCH_DIR, "inspect.ndjson");

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
  proposal: "Document-RAG/presentation/index.html; 开题报告.md",
};

const SLIDES = [
  {
    kicker: "多模态大模型原理与应用 · 题目 3",
    title: "文档理解与多模态检索问答系统",
    subtitle: "基于 Qwen2.5-VL 与 LangChain 的多模态文档问答设计，以轻量文档图增强作为进阶方向。",
    tags: ["多模态 RAG", "DocVQA / ChartQA", "轻量 GraphRAG", "课程作业开题展示"],
    footer: [
      ["展示目标", "讲清主线交付、增强目标、风险与回退"],
      ["当前口径", "最低可交付版本以多模态 RAG 为主"],
    ],
    notes: "封面只强调三件事：课程主线、增强方向、风险可控。",
    sources: ["proposal"],
  },
  {
    kicker: "01 · WHY THIS PROBLEM",
    title: "文档问答的难点不只在 OCR，而在证据组织",
    cards: [
      ["A", "文档天然多模态", "文本、图表、表格、版面结构混在一起。只看纯文本，很容易丢掉图像与版面中的关键信息。"],
      ["B", "问题常常跨页、跨块", "“图 3 说明了什么”这类问题并不只依赖单段文字，还可能关联图号、页面、章节结论和上下文引用。"],
      ["C", "仅靠向量相似度不够", "传统 RAG 能找到“像答案”的片段，但不擅长表达“这个片段为什么和那个图有关”。"],
    ],
    stripTitle: "核心研究问题",
    stripBody: "在先满足课程题目 3 最低要求的前提下，多模态信息能否显著提升文档问答效果；进一步引入轻量图结构后，是否能在多跳推理问题上提供额外收益？",
    notes: "问题定义要从工程困难与答辩可解释性两侧讲。",
    sources: ["proposal"],
  },
  {
    kicker: "02 · SCOPE CONTROL",
    title: "课程最低要求与项目目标分层",
    leftTitle: "课程最低验收",
    leftItems: [
      "支持上传多页 PDF 或图像",
      "自动分块文本 / 表格 / 图形内容",
      "支持自然语言查询并标注页面或图号引用",
      "至少在一个文档数据集上评测",
      "完成纯文本 RAG 与多模态 RAG 对比",
    ],
    rightBlocks: [
      ["最低可交付版本", "先做稳的多模态 RAG 闭环", [
        "页面解析 + 结构化分块",
        "FAISS 向量检索",
        "多模态问答与页码 / 图号引用",
        "纯文本 vs 多模态两组评测",
      ]],
      ["增强目标", "加入轻量文档图增强", [
        "规则边 + 可选语义边",
        "文本 GraphRAG / 多模态 GraphRAG",
        "多跳推理与关系路径引用探索",
      ]],
    ],
    notes: "这一页是整套报告的防守核心，要讲清主线与增强分层。",
    sources: ["proposal"],
  },
  {
    kicker: "03 · MAINLINE PIPELINE",
    title: "先交付课程要求，再谈增强模块",
    steps: [
      ["1", "上传与渲染", "PDF / 图像输入，经 PyMuPDF 渲染为页面图像。"],
      ["2", "结构化抽取", "用 Qwen2.5-VL 抽取文本、表格、图表描述与页码元数据。"],
      ["3", "向量化索引", "用 LangChain + FAISS 建立可检索的 Document 索引。"],
      ["4", "多模态问答", "检索文本块与原页面图像，共同送入 VLM 回答问题。"],
      ["5", "引用标注", "输出页码、图号等可追踪引用，满足题目 3 的答题要求。"],
    ],
    principles: [
      ["先闭环", "先保证上传、问答、引用、评测四件事跑通。"],
      ["再增强", "GraphRAG 只在主链路稳定之后增加，不抢主线资源。"],
      ["重答辩价值", "每个模块都要能解释“为什么做”和“如果没做成怎么办”。"],
    ],
    notes: "这一页要强调工程顺序，而不是模型名堆砌。",
    sources: ["proposal"],
  },
  {
    kicker: "04 · GRAPHRAG POSITIONING",
    title: "我们做的是轻量文档图增强，不是微软 GraphRAG 全复现",
    borrow: [
      "用图结构增强检索，而不是只靠向量相似度",
      "围绕问题构建局部证据上下文",
      "让多跳关系成为可解释的补充信号",
    ],
    boundary: [
      "不复现 community-based indexing",
      "不生成 community reports",
      "不做 global search 的全局主题总结",
      "不把轻量文档图表述成严格知识库",
    ],
    caption: "当前增强部分只面向具体问题的局部图检索：种子节点、邻居扩展、局部证据融合，而不是宏观主题总结。",
    notes: "答辩里要把“借鉴”和“不做”并列讲，主动收口。",
    sources: ["proposal"],
  },
  {
    kicker: "05 · DATASETS & EVALUATION",
    title: "实验分成必做对比与增强探索两层",
    datasets: [
      ["数据集 A", "学术论文 PDF", "3 份含图表、表格和跨节引用关系的论文，用于构建自建测试集。"],
      ["数据集 B", "ChartQA 子集", "重点验证图表理解能力，检查视觉信息是否真正带来收益。"],
      ["数据集 C", "DocVQA 子集", "作为结构化文档问答补充，增强评测的可比性与覆盖面。"],
    ],
    experiments: [
      ["必做实验", "纯文本 RAG vs 多模态 RAG", [
        "重点看图表题、表格题是否提升",
        "重点检查引用页码 / 图号是否稳定",
        "对应课程题目 3 的最低验收要求",
      ]],
      ["增强实验", "文本 GraphRAG / 多模态 GraphRAG", [
        "重点看多跳推理题是否受益",
        "重点分析关系路径是否更清晰",
        "若收益不足，可保留为探索性结果",
      ]],
    ],
    notes: "评测设计上必须把“必做”与“探索”分开说。",
    sources: ["proposal"],
  },
  {
    kicker: "06 · RISK CONTROL",
    title: "最重要的不是功能堆叠，而是主线一定可交付",
    risks: [
      ["结构化输出不稳定", "页面解析结果不统一，后续索引会被带偏。", "先保 texts / tables / figures 三类字段稳定。"],
      ["关系抽取噪声大", "同名实体、别名、缩写跨页难对齐，图谱可能误导回答。", "优先使用规则边，语义边只做增强。"],
      ["增强实验时间不够", "四组系统同时推进，容易让所有链路都不完整。", "优先保住两组对比、基础评测和展示界面。"],
      ["对照不够严格", "纯文本组与多模态组模型能力不完全等价。", "明确说明是工程比较，不夸成严格因果消融。"],
    ],
    notes: "这一页就是回退方案页，必须讲得稳。",
    sources: ["proposal"],
  },
  {
    kicker: "07 · DELIVERY PLAN",
    title: "我们希望交付的，不只是一个想法，而是一条可收敛的路线",
    contributions: [
      "先交付课程要求对应的多模态问答闭环",
      "把 GraphRAG 设计成增量增强，而非单点前提",
      "整理引用标注质量与失败案例，提升答辩可解释性",
      "在无本地 GPU 条件下完成可演示原型",
    ],
    phases: [
      ["阶段 1", "解析与分块", "PDF 渲染\n结构化抽取"],
      ["阶段 2", "索引与基线", "FAISS 索引\n纯文本基线"],
      ["阶段 3", "多模态 RAG", "问答闭环\n引用标注"],
      ["阶段 4", "图增强探索", "知识图增强\nGraphRAG"],
    ],
    caption: "先做解析、索引、问答、评测四步主线，再尝试图增强与多跳推理。",
    closing: "一句话总结：先做稳的多模态 RAG，证明题目 3 的主线能跑通；GraphRAG 作为增强方向，用来争取多跳推理和结构解释能力的增益。",
    notes: "最后一页要把“可交付路线”作为收束，不要收在技术名词上。",
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
  addText(slide, slideNo, "答辩重点", 856, 156, 120, 18, {
    size: 13,
    color: ACCENT_DEEP,
    bold: true,
    face: MONO_FACE,
    role: "cover summary label",
    checkFit: false,
  });
  addBulletList(slide, slideNo, [
    "最低可交付版本明确",
    "GraphRAG 只作为增强方向",
    "风险与回退方案可直接答辩",
  ], 856, 194, 258, 92, {
    size: 18,
    color: BODY,
    role: "cover summary list",
  });
  addShape(slide, slideNo, "rect", 856, 310, 230, 2, "#BFDAD2", "#00000000", 0, "cover summary rule");
  addText(slide, slideNo, "主线优先 · 增强有边界 · 结果可解释", 856, 332, 244, 34, {
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
  addText(slide, slideNo, "答辩口径", 856, 434, 80, 16, {
    size: 12,
    color: MUTE,
    bold: true,
    face: MONO_FACE,
    role: "cover lower label",
    checkFit: false,
  });
  addText(slide, slideNo, "先证明题目 3 主线跑通，再展示 GraphRAG 的增益空间。", 856, 460, 270, 78, {
    size: 19,
    color: ACCENT_DEEP,
    bold: true,
    face: TITLE_FACE,
    role: "hero note",
  });
  addText(slide, slideNo, "重点不是功能堆叠，而是路线可收敛、结果可解释。", 856, 556, 264, 34, {
    size: 14,
    color: BODY,
    role: "hero note caption",
  });

  addNotes(slide, data.notes, data.sources);
}

function slideProblem(presentation) {
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

function slideScope(presentation) {
  const slideNo = 3;
  const data = SLIDES[2];
  const slide = presentation.slides.add();
  addBackground(slide, slideNo);
  addHeader(slide, slideNo, data.kicker);
  addTitle(slide, slideNo, data.title);

  addPanel(slide, slideNo, 76, 246, 372, 392, { fill: SURFACE, line: BORDER, radiusRole: "scope left", accent: ACCENT });
  addText(slide, slideNo, data.leftTitle, 108, 276, 220, 22, {
    size: 15,
    color: ACCENT_DEEP,
    bold: true,
    role: "scope left title",
    checkFit: false,
  });
  addBulletList(slide, slideNo, data.leftItems, 108, 320, 304, 240, {
    size: 18,
    color: BODY,
    role: "scope left list",
  });

  const [block1, block2] = data.rightBlocks;
  addPanel(slide, slideNo, 478, 246, 720, 178, { fill: SURFACE, line: BORDER, radiusRole: "scope right block", accent: ACCENT });
  addPanel(slide, slideNo, 478, 446, 720, 192, { fill: SURFACE_ALT, line: BORDER, radiusRole: "scope right block", accent: ACCENT });

  addText(slide, slideNo, block1[0], 510, 270, 140, 18, { size: 13, color: ACCENT_DEEP, bold: true, role: "scope block label", checkFit: false });
  addText(slide, slideNo, block1[1], 510, 300, 336, 28, { size: 24, color: INK, bold: true, face: TITLE_FACE, role: "scope block title", checkFit: false });
  addBulletList(slide, slideNo, block1[2], 510, 344, 628, 84, { size: 17, color: BODY, role: "scope block list" });

  addText(slide, slideNo, block2[0], 510, 470, 140, 18, { size: 13, color: ACCENT_DEEP, bold: true, role: "scope block label", checkFit: false });
  addText(slide, slideNo, block2[1], 510, 500, 368, 28, { size: 24, color: INK, bold: true, face: TITLE_FACE, role: "scope block title", checkFit: false });
  addBulletList(slide, slideNo, block2[2], 510, 544, 628, 76, { size: 17, color: BODY, role: "scope block list" });

  addNotes(slide, data.notes, data.sources);
}

function slidePipeline(presentation) {
  const slideNo = 4;
  const data = SLIDES[3];
  const slide = presentation.slides.add();
  addBackground(slide, slideNo);
  addHeader(slide, slideNo, data.kicker);
  addTitle(slide, slideNo, data.title);

  addShape(slide, slideNo, "rect", 138, 316, 1004, 6, "#CFE6E0", "#00000000", 0, "pipeline line");
  for (let i = 0; i < data.steps.length; i += 1) {
    const [num, title, body] = data.steps[i];
    const cx = 136 + i * 248;
    addShape(slide, slideNo, "ellipse", cx, 286, 64, 64, ACCENT, "#00000000", 0, "pipeline node");
    addText(slide, slideNo, num, cx + 24, 308, 16, 18, {
      size: 20,
      color: SURFACE,
      bold: true,
      role: "pipeline node text",
      checkFit: false,
    });
    addText(slide, slideNo, title, cx - 20, 362, 108, 24, {
      size: 18,
      color: INK,
      bold: true,
      role: "pipeline title",
      align: "center",
      checkFit: false,
    });
    addText(slide, slideNo, body, cx - 44, 394, 156, 64, {
      size: 14,
      color: BODY,
      role: "pipeline body",
      align: "center",
    });
  }

  const cardY = 532;
  const cardW = 354;
  for (let i = 0; i < data.principles.length; i += 1) {
    const [title, body] = data.principles[i];
    const x = 76 + i * (cardW + 22);
    addPanel(slide, slideNo, x, cardY, cardW, 114, { fill: i === 1 ? SURFACE_ALT : SURFACE, line: BORDER, radiusRole: "principle card" });
    addText(slide, slideNo, title, x + 26, cardY + 22, 160, 22, {
      size: 22,
      color: ACCENT_DEEP,
      bold: true,
      face: TITLE_FACE,
      role: "principle title",
      checkFit: false,
    });
    addText(slide, slideNo, body, x + 26, cardY + 56, 300, 36, {
      size: 15,
      color: BODY,
      role: "principle body",
    });
  }

  addNotes(slide, data.notes, data.sources);
}

function drawGraphDiagram(slide, slideNo) {
  addPanel(slide, slideNo, 782, 258, 394, 314, { fill: SURFACE_ALT, line: BORDER, radiusRole: "graph panel" });
  const nodes = [
    [938, 302, 72, "问题"],
    [848, 402, 72, "图表"],
    [1026, 402, 72, "页码"],
    [886, 486, 72, "段落"],
    [988, 486, 72, "章节"],
  ];
  const lines = [
    [974, 372, 884, 432],
    [974, 372, 1062, 432],
    [884, 436, 922, 500],
    [1062, 436, 1024, 500],
    [922, 522, 988, 522],
  ];
  for (const [x1, y1, x2, y2] of lines) {
    const left = Math.min(x1, x2);
    const top = Math.min(y1, y2);
    const width = Math.max(2, Math.abs(x2 - x1));
    const height = Math.max(2, Math.abs(y2 - y1));
    addShape(slide, slideNo, "rect", left, top, width, height < 8 ? 4 : 4, "#B7D9D1", "#00000000", 0, "graph edge");
  }
  for (const [x, y, size, label] of nodes) {
    addShape(slide, slideNo, "ellipse", x, y, size, size, label === "问题" ? ACCENT : SURFACE, ACCENT, 1.4, "graph node");
    addText(slide, slideNo, label, x + 12, y + 24, size - 24, 18, {
      size: 16,
      color: label === "问题" ? SURFACE : ACCENT_DEEP,
      bold: true,
      role: "graph node label",
      align: "center",
      checkFit: false,
    });
  }
}

function slideGraph(presentation) {
  const slideNo = 5;
  const data = SLIDES[4];
  const slide = presentation.slides.add();
  addBackground(slide, slideNo, "warm");
  addHeader(slide, slideNo, data.kicker);
  addTitle(slide, slideNo, data.title, null, 74, 90, 990);

  addPanel(slide, slideNo, 76, 240, 324, 238, { fill: SURFACE, line: BORDER, radiusRole: "borrow panel", accent: ACCENT });
  addPanel(slide, slideNo, 424, 240, 324, 238, { fill: SURFACE, line: BORDER, radiusRole: "boundary panel", accent: ACCENT });
  addText(slide, slideNo, "借鉴什么", 104, 268, 120, 22, { size: 22, color: INK, bold: true, face: TITLE_FACE, role: "borrow title", checkFit: false });
  addText(slide, slideNo, "明确不做什么", 452, 268, 160, 22, { size: 22, color: INK, bold: true, face: TITLE_FACE, role: "boundary title", checkFit: false });
  addBulletList(slide, slideNo, data.borrow, 104, 314, 260, 116, { size: 17, color: BODY, role: "borrow list" });
  addBulletList(slide, slideNo, data.boundary, 452, 314, 260, 150, { size: 17, color: BODY, role: "boundary list" });

  drawGraphDiagram(slide, slideNo);
  addText(slide, slideNo, data.caption, 782, 586, 394, 48, {
    size: 15,
    color: BODY,
    role: "graph caption",
  });

  addNotes(slide, data.notes, data.sources);
}

function slideEval(presentation) {
  const slideNo = 6;
  const data = SLIDES[5];
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
    addPanel(slide, slideNo, x, experimentY, 542, 190, {
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
    addBulletList(slide, slideNo, items, x + 28, experimentY + 116, 470, 72, {
      size: 17,
      color: BODY,
      role: "experiment list",
    });
  }

  addNotes(slide, data.notes, data.sources);
}

function slideRisk(presentation) {
  const slideNo = 7;
  const data = SLIDES[6];
  const slide = presentation.slides.add();
  addBackground(slide, slideNo, "warm");
  addHeader(slide, slideNo, data.kicker);
  addTitle(slide, slideNo, data.title);

  const cardW = 540;
  const cardH = 166;
  for (let i = 0; i < data.risks.length; i += 1) {
    const [title, body, fallback] = data.risks[i];
    const row = Math.floor(i / 2);
    const col = i % 2;
    const x = 76 + col * (cardW + 24);
    const y = 246 + row * (cardH + 22);
    addPanel(slide, slideNo, x, y, cardW, cardH, { fill: SURFACE, line: BORDER, radiusRole: "risk card" });
    addText(slide, slideNo, title, x + 24, y + 22, 240, 22, {
      size: 22,
      color: INK,
      bold: true,
      face: TITLE_FACE,
      role: "risk title",
      checkFit: false,
    });
    addText(slide, slideNo, body, x + 24, y + 62, 470, 34, {
      size: 16,
      color: BODY,
      role: "risk body",
    });
    addShape(slide, slideNo, "roundRect", x + 24, y + 112, 480, 34, WARM, "#00000000", 0, "risk fallback bg");
    addText(slide, slideNo, fallback, x + 38, y + 120, 446, 16, {
      size: 14,
      color: ACCENT_DEEP,
      bold: true,
      role: "risk fallback",
      checkFit: false,
    });
  }

  addNotes(slide, data.notes, data.sources);
}

function slideDelivery(presentation) {
  const slideNo = 8;
  const data = SLIDES[7];
  const slide = presentation.slides.add();
  addBackground(slide, slideNo);
  addHeader(slide, slideNo, data.kicker);
  addTitle(slide, slideNo, data.title, null, 74, 90, 980);

  addPanel(slide, slideNo, 76, 246, 444, 286, { fill: SURFACE, line: BORDER, radiusRole: "contribution panel", accent: ACCENT });
  addText(slide, slideNo, "预期贡献", 104, 272, 140, 20, {
    size: 15,
    color: ACCENT_DEEP,
    bold: true,
    role: "contribution label",
    checkFit: false,
  });
  addBulletList(slide, slideNo, data.contributions, 104, 316, 378, 164, {
    size: 18,
    color: BODY,
    role: "contribution list",
  });

  addPanel(slide, slideNo, 548, 246, 650, 286, { fill: SURFACE_ALT, line: BORDER, radiusRole: "timeline panel", accent: ACCENT });
  addText(slide, slideNo, "阶段计划", 576, 272, 120, 20, {
    size: 15,
    color: ACCENT_DEEP,
    bold: true,
    role: "timeline label",
    checkFit: false,
  });

  addShape(slide, slideNo, "rect", 650, 396, 430, 5, "#C6E1DA", "#00000000", 0, "timeline line");
  for (let i = 0; i < data.phases.length; i += 1) {
    const [phase, title, detail] = data.phases[i];
    const cx = 662 + i * 128;
    addShape(slide, slideNo, "ellipse", cx, 368, 56, 56, i < 3 ? ACCENT : "#DCE7E3", "#00000000", 0, "timeline node");
    addText(slide, slideNo, String(i + 1), cx + 21, 388, 14, 16, {
      size: 18,
      color: i < 3 ? SURFACE : BODY,
      bold: true,
      role: "timeline node number",
      checkFit: false,
    });
    addText(slide, slideNo, phase, cx - 8, 326, 72, 18, {
      size: 14,
      color: ACCENT_DEEP,
      bold: true,
      role: "timeline phase",
      align: "center",
      checkFit: false,
    });
    addText(slide, slideNo, title, cx - 26, 344, 108, 18, {
      size: 16,
      color: INK,
      bold: true,
      role: "timeline title",
      align: "center",
      checkFit: false,
    });
    addShape(slide, slideNo, "roundRect", cx - 40, 444, 136, 54, SURFACE, "#00000000", 0, "timeline detail bg");
    addText(slide, slideNo, detail, cx - 26, 456, 108, 30, {
      size: 13,
      color: BODY,
      role: "timeline detail",
      align: "center",
    });
  }

  addText(slide, slideNo, data.caption, 576, 552, 566, 34, {
    size: 16,
    color: BODY,
    role: "timeline caption",
  });

  addPanel(slide, slideNo, 76, 572, 1122, 72, { fill: SURFACE, line: BORDER, radiusRole: "closing panel" });
  addText(slide, slideNo, data.closing, 104, 596, 1060, 24, {
    size: 16,
    color: INK,
    bold: true,
    role: "closing text",
  });

  addNotes(slide, data.notes, data.sources);
}

function createDeck() {
  const presentation = Presentation.create({ slideSize: { width: W, height: H } });
  slideCover(presentation);
  slideProblem(presentation);
  slideScope(presentation);
  slidePipeline(presentation);
  slideGraph(presentation);
  slideEval(presentation);
  slideRisk(presentation);
  slideDelivery(presentation);
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
  const pptxPath = path.join(OUT_DIR, "output.pptx");
  await pptxBlob.save(pptxPath);
  return pptxPath;
}

const presentation = createDeck();
const pptxPath = await verifyAndExport(presentation);
console.log(pptxPath);
