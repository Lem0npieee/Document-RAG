const PptxGenJS = require("pptxgenjs");

const pres = new PptxGenJS();
pres.layout = "LAYOUT_16x9";
pres.author = "QClaw";
pres.title = "文档理解与多模态 GraphRAG 检索问答系统";
pres.subject = "开题报告答辩 PPT";

// ========== 配色方案：深蓝科技风 ==========
const C = {
  primary: "0A2540",    // 深海蓝
  secondary: "1B4965",  // 钢蓝
  accent: "00D4AA",     // 科技青
  light: "E8F4F8",      // 浅冰蓝
  white: "FFFFFF",
  darkText: "1A1A2E",
  muted: "6B7B8C",
};

// ========== 辅助函数 ==========
function addTextBox(slide, text, opts) {
  slide.addText(text, opts);
}

function makeShadow() {
  return { type: "outer", color: "000000", blur: 8, offset: 3, angle: 135, opacity: 0.12 };
}

// ========== Slide 1: 封面 ==========
(function () {
  const s = pres.addSlide();
  s.background = { color: C.primary };

  // 顶部装饰条
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: "100%", h: 0.08,
    fill: { color: C.accent }, line: { color: C.accent, width: 0 }
  });

  // 左侧竖条
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.8, w: 0.06, h: 2.8,
    fill: { color: C.accent }, line: { color: C.accent, width: 0 }
  });

  // 主标题
  s.addText("文档理解与多模态 GraphRAG\n检索问答系统", {
    x: 0.8, y: 1.6, w: 8.5, h: 1.6,
    fontSize: 40, fontFace: "Microsoft YaHei", bold: true,
    color: C.white, align: "left", valign: "middle",
    lineSpacing: 38
  });

  // 副标题
  s.addText("基于 qwen3-vl-8b-instruct、FAISS 与 networkx 的可追溯文档智能助手", {
    x: 0.8, y: 3.4, w: 8.5, h: 0.6,
    fontSize: 16, fontFace: "Microsoft YaHei",
    color: C.accent, align: "left", valign: "middle"
  });

  // 底部信息
  s.addText("开题报告答辩", {
    x: 0.8, y: 4.6, w: 3, h: 0.4,
    fontSize: 14, fontFace: "Microsoft YaHei",
    color: "A0B4C4", align: "left"
  });

  s.addText("2025", {
    x: 0.8, y: 5.0, w: 2, h: 0.3,
    fontSize: 12, fontFace: "Microsoft YaHei",
    color: "7A8FA0", align: "left"
  });
})();

// ========== Slide 2: 任务背景 ==========
(function () {
  const s = pres.addSlide();
  s.background = { color: C.white };

  // 顶部色块
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: "100%", h: 1.1,
    fill: { color: C.primary }, line: { color: C.primary, width: 0 }
  });

  s.addText("任务背景", {
    x: 0.5, y: 0.25, w: 4, h: 0.6,
    fontSize: 32, fontFace: "Microsoft YaHei", bold: true,
    color: C.white, align: "left", valign: "middle"
  });

  // 核心问题卡片背景
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.5, w: 9, h: 1.2,
    fill: { color: C.light }, line: { color: "D0E4EC", width: 1 },
    shadow: makeShadow()
  });

  s.addText("核心问题", {
    x: 0.7, y: 1.6, w: 1.5, h: 0.35,
    fontSize: 14, fontFace: "Microsoft YaHei", bold: true,
    color: C.accent, align: "left"
  });

  s.addText("对于图文混排、跨页引用和结构复杂的长文档，如何把页面视觉信息、文本语义片段和知识图谱关系融合到同一条检索问答链路中？", {
    x: 0.7, y: 1.95, w: 8.6, h: 0.6,
    fontSize: 15, fontFace: "Microsoft YaHei",
    color: C.darkText, align: "left", valign: "top",
    lineSpacing: 22
  });

  // 三个痛点
  const painPoints = [
    { title: "视觉信息丢失", desc: "图表趋势、版面结构、颜色标记、坐标轴和表格边界很难被纯文本准确保留" },
    { title: "结构关系不足", desc: "同一页内的图文对应、跨页段落延续、图号与结论之间的支撑关系无法仅靠向量相似度稳定表达" },
    { title: "证据链不可解释", desc: "向量检索可以找到相似片段，但不擅长回答“某结论由哪些图表和实验支撑”这类多跳问题" }
  ];

  painPoints.forEach((p, i) => {
    const x = 0.5 + i * 3.2;
    s.addShape(pres.shapes.RECTANGLE, {
      x: x, y: 3.0, w: 2.9, h: 2.2,
      fill: { color: C.white }, line: { color: "E2E8F0", width: 1 },
      shadow: makeShadow()
    });

    // 小色块
    s.addShape(pres.shapes.RECTANGLE, {
      x: x + 0.15, y: 3.15, w: 0.5, h: 0.08,
      fill: { color: C.accent }, line: { color: C.accent, width: 0 }
    });

    s.addText(p.title, {
      x: x + 0.15, y: 3.35, w: 2.6, h: 0.4,
      fontSize: 16, fontFace: "Microsoft YaHei", bold: true,
      color: C.primary, align: "left"
    });

    s.addText(p.desc, {
      x: x + 0.15, y: 3.85, w: 2.6, h: 1.2,
      fontSize: 12, fontFace: "Microsoft YaHei",
      color: C.muted, align: "left", valign: "top",
      lineSpacing: 18
    });
  });
})();

// ========== Slide 3: 系统架构图 ==========
(function () {
  const s = pres.addSlide();
  s.background = { color: C.white };

  // 顶部色块
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: "100%", h: 1.1,
    fill: { color: C.primary }, line: { color: C.primary, width: 0 }
  });

  s.addText("系统架构", {
    x: 0.5, y: 0.25, w: 4, h: 0.6,
    fontSize: 32, fontFace: "Microsoft YaHei", bold: true,
    color: C.white, align: "left", valign: "middle"
  });

  // 架构图
  s.addImage({
    path: "DocRAG架构.png",
    x: 0.5, y: 1.4, w: 9, h: 3.8,
    sizing: { type: "contain", w: 9, h: 3.8 }
  });

  // 底部说明
  s.addText("知识库构建链路（左）+ 在线问答链路（右）+ 前端展示层（下）", {
    x: 0.5, y: 5.3, w: 9, h: 0.3,
    fontSize: 12, fontFace: "Microsoft YaHei",
    color: C.muted, align: "center"
  });
})();

// ========== Slide 4: 实验设计 ==========
(function () {
  const s = pres.addSlide();
  s.background = { color: C.white };

  // 顶部色块
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: "100%", h: 1.1,
    fill: { color: C.primary }, line: { color: C.primary, width: 0 }
  });

  s.addText("实验设计", {
    x: 0.5, y: 0.25, w: 4, h: 0.6,
    fontSize: 32, fontFace: "Microsoft YaHei", bold: true,
    color: C.white, align: "left", valign: "middle"
  });

  // 左侧：数据集
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.4, w: 4.2, h: 3.8,
    fill: { color: C.light }, line: { color: "D0E4EC", width: 1 },
    shadow: makeShadow()
  });

  s.addText("数据集", {
    x: 0.7, y: 1.55, w: 2, h: 0.4,
    fontSize: 18, fontFace: "Microsoft YaHei", bold: true,
    color: C.primary, align: "left"
  });

  const datasets = [
    "自建长文档测试集（3-5 篇论文/技术文档）",
    "DocVQA 验证子集（单页文档问答）",
    "ChartQA 子集（图表推理问题）"
  ];
  datasets.forEach((d, i) => {
    s.addText(d, {
      x: 0.7, y: 2.1 + i * 0.55, w: 3.8, h: 0.45,
      fontSize: 13, fontFace: "Microsoft YaHei",
      color: C.darkText, align: "left", valign: "middle"
    });
    s.addShape(pres.shapes.RECTANGLE, {
      x: 0.7, y: 2.1 + i * 0.55 + 0.18, w: 0.12, h: 0.12,
      fill: { color: C.accent }, line: { color: C.accent, width: 0 }
    });
  });

  // 右侧：评测指标
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.4, w: 4.3, h: 3.8,
    fill: { color: C.white }, line: { color: "E2E8F0", width: 1 },
    shadow: makeShadow()
  });

  s.addText("评测指标", {
    x: 5.4, y: 1.55, w: 2, h: 0.4,
    fontSize: 18, fontFace: "Microsoft YaHei", bold: true,
    color: C.primary, align: "left"
  });

  const metrics = [
    { name: "ANLS", desc: "平均归一化编辑相似度" },
    { name: "Exact Match", desc: "预测与标准答案完全匹配" },
    { name: "引用命中率", desc: "页码/图号/节点覆盖真实证据" },
    { name: "证据召回率", desc: "top-k + 图扩展包含标注依据" },
    { name: "图谱有效性", desc: "开启/关闭图扩展的多跳题差异" }
  ];
  metrics.forEach((m, i) => {
    s.addText(m.name, {
      x: 5.4, y: 2.1 + i * 0.55, w: 1.8, h: 0.3,
      fontSize: 13, fontFace: "Microsoft YaHei", bold: true,
      color: C.accent, align: "left"
    });
    s.addText(m.desc, {
      x: 7.3, y: 2.1 + i * 0.55, w: 2, h: 0.3,
      fontSize: 12, fontFace: "Microsoft YaHei",
      color: C.muted, align: "left"
    });
  });

  // 底部：消融实验
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 5.4, w: 9, h: 0.5,
    fill: { color: C.secondary }, line: { color: C.secondary, width: 0 }
  });

  s.addText("消融实验：Full（完整系统） / No-Graph（关闭图谱扩展） / No-Image（关闭页面原图）", {
    x: 0.5, y: 5.4, w: 9, h: 0.5,
    fontSize: 13, fontFace: "Microsoft YaHei",
    color: C.white, align: "center", valign: "middle"
  });
})();

// ========== Slide 5: Demo 展示 ==========
(function () {
  const s = pres.addSlide();
  s.background = { color: C.white };

  // 顶部色块
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: "100%", h: 1.1,
    fill: { color: C.primary }, line: { color: C.primary, width: 0 }
  });

  s.addText("系统 Demo 展示", {
    x: 0.5, y: 0.25, w: 5, h: 0.6,
    fontSize: 32, fontFace: "Microsoft YaHei", bold: true,
    color: C.white, align: "left", valign: "middle"
  });

  // 四个功能模块
  const demos = [
    { title: "Chat Workspace", desc: "用户提问与模型回答交互界面" },
    { title: "Knowledge Graph", desc: "vis-network 可视化文档图谱" },
    { title: "Node Inspector", desc: "节点内容、页码、bbox 局部预览" },
    { title: "上传与入库", desc: "PDF/图片上传、解析、索引一键完成" }
  ];

  demos.forEach((d, i) => {
    const row = Math.floor(i / 2);
    const col = i % 2;
    const x = 0.5 + col * 4.7;
    const y = 1.5 + row * 2.1;

    s.addShape(pres.shapes.RECTANGLE, {
      x: x, y: y, w: 4.4, h: 1.8,
      fill: { color: C.white }, line: { color: "E2E8F0", width: 1 },
      shadow: makeShadow()
    });

    // 左侧色条
    s.addShape(pres.shapes.RECTANGLE, {
      x: x, y: y, w: 0.08, h: 1.8,
      fill: { color: C.accent }, line: { color: C.accent, width: 0 }
    });

    s.addText(d.title, {
      x: x + 0.25, y: y + 0.2, w: 3.8, h: 0.4,
      fontSize: 18, fontFace: "Microsoft YaHei", bold: true,
      color: C.primary, align: "left"
    });

    s.addText(d.desc, {
      x: x + 0.25, y: y + 0.7, w: 3.8, h: 0.8,
      fontSize: 13, fontFace: "Microsoft YaHei",
      color: C.muted, align: "left", valign: "top",
      lineSpacing: 20
    });
  });

  // 底部提示
  s.addText("前端基于原生 HTML/CSS/JS + vis-network 构建，支持图谱交互与节点详情查看", {
    x: 0.5, y: 5.4, w: 9, h: 0.3,
    fontSize: 12, fontFace: "Microsoft YaHei",
    color: C.muted, align: "center"
  });
})();

// ========== Slide 6: 算力预算与时间计划 ==========
(function () {
  const s = pres.addSlide();
  s.background = { color: C.white };

  // 顶部色块
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: "100%", h: 1.1,
    fill: { color: C.primary }, line: { color: C.primary, width: 0 }
  });

  s.addText("算力预算与时间计划", {
    x: 0.5, y: 0.25, w: 6, h: 0.6,
    fontSize: 32, fontFace: "Microsoft YaHei", bold: true,
    color: C.white, align: "left", valign: "middle"
  });

  // 左侧：算力预算
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 1.4, w: 4.2, h: 3.6,
    fill: { color: C.light }, line: { color: "D0E4EC", width: 1 },
    shadow: makeShadow()
  });

  s.addText("算力与成本", {
    x: 0.7, y: 1.55, w: 2.5, h: 0.4,
    fontSize: 18, fontFace: "Microsoft YaHei", bold: true,
    color: C.primary, align: "left"
  });

  const resources = [
    { item: "本地计算", val: "CPU 即可" },
    { item: "GPU", val: "非必需（API 调用）" },
    { item: "存储", val: "百 MB ~ 数 GB" },
    { item: "API 成本", val: "与页数/问题数相关" },
    { item: "网络", val: "构建和问答时需要" }
  ];
  resources.forEach((r, i) => {
    s.addText(r.item, {
      x: 0.7, y: 2.1 + i * 0.55, w: 1.5, h: 0.35,
      fontSize: 13, fontFace: "Microsoft YaHei", bold: true,
      color: C.darkText, align: "left"
    });
    s.addText(r.val, {
      x: 2.3, y: 2.1 + i * 0.55, w: 2.2, h: 0.35,
      fontSize: 13, fontFace: "Microsoft YaHei",
      color: C.muted, align: "left"
    });
  });

  // 右侧：时间计划
  s.addShape(pres.shapes.RECTANGLE, {
    x: 5.2, y: 1.4, w: 4.3, h: 3.6,
    fill: { color: C.white }, line: { color: "E2E8F0", width: 1 },
    shadow: makeShadow()
  });

  s.addText("后续计划", {
    x: 5.4, y: 1.55, w: 2.5, h: 0.4,
    fontSize: 18, fontFace: "Microsoft YaHei", bold: true,
    color: C.primary, align: "left"
  });

  const timeline = [
    { week: "第 1 周", task: "清理图谱噪声、优化关键词过滤" },
    { week: "第 2 周", task: "构建自建测试集、补充 DocVQA" },
    { week: "第 3 周", task: "消融实验：Full / No-Graph / No-Image" },
    { week: "第 4 周", task: "完善前端、整理报告与答辩材料" }
  ];
  timeline.forEach((t, i) => {
    // 圆点
    s.addShape(pres.shapes.OVAL, {
      x: 5.5, y: 2.15 + i * 0.7, w: 0.18, h: 0.18,
      fill: { color: C.accent }, line: { color: C.accent, width: 0 }
    });

    s.addText(t.week, {
      x: 5.8, y: 2.1 + i * 0.7, w: 1.2, h: 0.3,
      fontSize: 13, fontFace: "Microsoft YaHei", bold: true,
      color: C.accent, align: "left"
    });

    s.addText(t.task, {
      x: 5.8, y: 2.4 + i * 0.7, w: 3.5, h: 0.3,
      fontSize: 12, fontFace: "Microsoft YaHei",
      color: C.darkText, align: "left"
    });

    if (i < timeline.length - 1) {
      s.addShape(pres.shapes.LINE, {
        x: 5.59, y: 2.38 + i * 0.7, w: 0, h: 0.45,
        line: { color: "D0E4EC", width: 1.5, dashType: "dash" }
      });
    }
  });

  // 底部状态
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0.5, y: 5.2, w: 9, h: 0.5,
    fill: { color: C.secondary }, line: { color: C.secondary, width: 0 }
  });

  s.addText("当前进度：工程骨架、文档解析、向量索引、图谱构建、GraphRAG 问答、Web 演示、评测框架 均已完成初版", {
    x: 0.5, y: 5.2, w: 9, h: 0.5,
    fontSize: 12, fontFace: "Microsoft YaHei",
    color: C.white, align: "center", valign: "middle"
  });
})();

// ========== Slide 7: 参考文献 ==========
(function () {
  const s = pres.addSlide();
  s.background = { color: C.white };

  // 顶部色块
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: "100%", h: 1.1,
    fill: { color: C.primary }, line: { color: C.primary, width: 0 }
  });

  s.addText("参考文献", {
    x: 0.5, y: 0.25, w: 4, h: 0.6,
    fontSize: 32, fontFace: "Microsoft YaHei", bold: true,
    color: C.white, align: "left", valign: "middle"
  });

  const refs = [
    "[1] Lewis, P. et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks. NeurIPS 2020.",
    "[2] Edge, D. et al. (2024). From Local to Global: A Graph RAG Approach to Query-Focused Summarization. arXiv:2404.16130.",
    "[3] Microsoft GraphRAG Documentation. Query Engine & Indexing Overview.",
    "[4] Mathew, M. et al. (2021). DocVQA: A Dataset for VQA on Document Images. WACV 2021.",
    "[5] Masry, A. et al. (2022). ChartQA: A Benchmark for QA about Charts. Findings of ACL 2022.",
    "[6] Faysse, M. et al. (2024). ColPali: Efficient Document Retrieval with Vision Language Models. arXiv:2407.01449.",
    "[7] Qwen Team. Qwen3-VL official repository & model card.",
    "[8] Dong, K. et al. (2025). MMDocRAG: Benchmarking Retrieval-Augmented Multimodal Generation for Document QA.",
    "[9] Bu, C. et al. (2025). Query-Driven Multimodal GraphRAG. Findings of ACL 2025."
  ];

  refs.forEach((r, i) => {
    s.addText(r, {
      x: 0.6, y: 1.4 + i * 0.45, w: 8.8, h: 0.4,
      fontSize: 11, fontFace: "Microsoft YaHei",
      color: C.darkText, align: "left", valign: "top",
      lineSpacing: 16
    });
  });
})();

// ========== Slide 8: 结束页 ==========
(function () {
  const s = pres.addSlide();
  s.background = { color: C.primary };

  // 顶部装饰条
  s.addShape(pres.shapes.RECTANGLE, {
    x: 0, y: 0, w: "100%", h: 0.08,
    fill: { color: C.accent }, line: { color: C.accent, width: 0 }
  });

  // 中央装饰线
  s.addShape(pres.shapes.LINE, {
    x: 3.5, y: 2.2, w: 3, h: 0,
    line: { color: C.accent, width: 2 }
  });

  s.addText("感谢聆听", {
    x: 1, y: 2.5, w: 8, h: 1,
    fontSize: 48, fontFace: "Microsoft YaHei", bold: true,
    color: C.white, align: "center", valign: "middle"
  });

  s.addText("欢迎提问与建议", {
    x: 1, y: 3.6, w: 8, h: 0.5,
    fontSize: 18, fontFace: "Microsoft YaHei",
    color: C.accent, align: "center", valign: "middle"
  });

  // 底部装饰线
  s.addShape(pres.shapes.LINE, {
    x: 3.5, y: 4.3, w: 3, h: 0,
    line: { color: C.accent, width: 2 }
  });
})();

// ========== 输出 ==========
pres.writeFile({ fileName: "开题报告答辩.pptx" })
  .then(() => console.log("PPT 生成成功：开题报告答辩.pptx"))
  .catch((err) => console.error("生成失败:", err));
