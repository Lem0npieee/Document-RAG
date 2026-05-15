const chatHistory = document.getElementById("chatHistory");
const chatInput = document.getElementById("chatInput");
const sendBtn = document.getElementById("sendBtn");
const fileInput = document.getElementById("fileInput");
const resetGraph = document.getElementById("resetGraph");
const centerGraph = document.getElementById("centerGraph");
const nodeInspector = document.getElementById("nodeInspector");
const nodeCount = document.getElementById("nodeCount");
const edgeCount = document.getElementById("edgeCount");
const graphCanvas = document.getElementById("graphCanvas");
const API_BASE = "";

const colorMap = {
  page: "#94a3b8",
  text: "#fcd34d",
  table: "#34d399",
  figure: "#f472b6",
  keyword: "#2591e9",
  cross_page_text: "#fb7185",
  conclusion: "#a78bfa",
  default: "#e2e8f0",
};

let network = null;
let graphState = { nodes: [], edges: [] };

const demoGraph = {
  nodes: [
    { id: "doc_text_1", label: "Section 1", type: "text", tokens: 120 },
    { id: "doc_table_1", label: "Table 2", type: "table", tokens: 60 },
    { id: "doc_fig_1", label: "Figure 3", type: "figure", tokens: 80 },
    { id: "keyword_qwen", label: "Qwen-VL", type: "keyword", tokens: 40 },
    { id: "keyword_acc", label: "Accuracy", type: "keyword", tokens: 30 },
  ],
  edges: [
    { from: "doc_text_1", to: "doc_table_1", relation: "鍚岄〉" },
    { from: "doc_table_1", to: "doc_fig_1", relation: "鏀拺缁撹" },
    { from: "doc_fig_1", to: "keyword_acc", relation: "shows" },
    { from: "keyword_qwen", to: "doc_fig_1", relation: "best_performance" },
  ],
};

function addMessage(role, text) {
  const bubble = document.createElement("div");
  bubble.className = `chat-bubble ${role}`;
  if (role === "assistant") {
    bubble.innerHTML = renderMarkdown(text);
  } else {
    bubble.textContent = text;
  }
  chatHistory.appendChild(bubble);
  chatHistory.scrollTop = chatHistory.scrollHeight;
}

function escapeHtml(value) {
  return String(value ?? "")
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

function renderInlineMarkdown(value) {
  const codeSpans = [];
  const placeholder = "\u0000CODE";
  let html = escapeHtml(value).replace(/`([^`]+)`/g, (_, code) => {
    codeSpans.push(`<code>${code}</code>`);
    return `${placeholder}${codeSpans.length - 1}\u0000`;
  });

  html = html
    .replace(/\*\*([^*\n]+)\*\*/g, "<strong>$1</strong>")
    .replace(/__([^_\n]+)__/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\n]+)\*/g, "$1<em>$2</em>")
    .replace(/(^|[^_])_([^_\n]+)_/g, "$1<em>$2</em>")
    .replace(/~~([^~\n]+)~~/g, "<del>$1</del>");

  const codePattern = new RegExp(`${placeholder}(\\d+)\\u0000`, "g");
  return html.replace(codePattern, (_, index) => codeSpans[Number(index)] || "");
}

function renderMarkdown(value) {
  const lines = String(value ?? "").replace(/\r\n?/g, "\n").split("\n");
  const blocks = [];
  let i = 0;

  const isBlockStart = (line) =>
    /^```/.test(line) ||
    /^#{1,4}\s+/.test(line) ||
    /^>\s?/.test(line) ||
    /^\s*([-*+])\s+/.test(line) ||
    /^\s*\d+\.\s+/.test(line);

  while (i < lines.length) {
    const line = lines[i];
    if (!line.trim()) {
      i += 1;
      continue;
    }

    if (/^```/.test(line)) {
      i += 1;
      const codeLines = [];
      while (i < lines.length && !/^```/.test(lines[i])) {
        codeLines.push(lines[i]);
        i += 1;
      }
      if (i < lines.length) i += 1;
      blocks.push(`<pre><code>${escapeHtml(codeLines.join("\n"))}</code></pre>`);
      continue;
    }

    const heading = line.match(/^(#{1,4})\s+(.+)$/);
    if (heading) {
      const level = Math.min(heading[1].length + 2, 6);
      blocks.push(`<h${level}>${renderInlineMarkdown(heading[2].trim())}</h${level}>`);
      i += 1;
      continue;
    }

    if (/^>\s?/.test(line)) {
      const quoteLines = [];
      while (i < lines.length && /^>\s?/.test(lines[i])) {
        quoteLines.push(lines[i].replace(/^>\s?/, ""));
        i += 1;
      }
      blocks.push(`<blockquote>${quoteLines.map(renderInlineMarkdown).join("<br>")}</blockquote>`);
      continue;
    }

    const unordered = line.match(/^\s*([-*+])\s+(.+)$/);
    const ordered = line.match(/^\s*\d+\.\s+(.+)$/);
    if (unordered || ordered) {
      const orderedList = Boolean(ordered);
      const tag = orderedList ? "ol" : "ul";
      const items = [];
      while (i < lines.length) {
        const item = orderedList
          ? lines[i].match(/^\s*\d+\.\s+(.+)$/)
          : lines[i].match(/^\s*[-*+]\s+(.+)$/);
        if (!item) break;
        items.push(`<li>${renderInlineMarkdown(item[1].trim())}</li>`);
        i += 1;
      }
      blocks.push(`<${tag}>${items.join("")}</${tag}>`);
      continue;
    }

    const paragraphLines = [];
    while (i < lines.length && lines[i].trim() && !isBlockStart(lines[i])) {
      paragraphLines.push(lines[i].trim());
      i += 1;
    }
    blocks.push(`<p>${paragraphLines.map(renderInlineMarkdown).join("<br>")}</p>`);
  }

  return blocks.join("");
}

function normalizeImageUrl(value) {
  const raw = String(value || "").replace(/\\/g, "/").trim();
  if (!raw) return "";
  if (raw.startsWith("http://") || raw.startsWith("https://") || raw.startsWith("/")) {
    return raw;
  }

  const lowered = raw.toLowerCase();
  const marker = "outputs/";
  const idx = lowered.indexOf(marker);
  if (idx !== -1) {
    return `/${raw.slice(idx)}`;
  }
  if (raw.startsWith("pages/")) {
    return `/outputs/${raw}`;
  }
  return "";
}

function normalizeBbox(value) {
  if (!Array.isArray(value) || value.length !== 4) return null;
  const nums = value.map((x) => Number(x));
  if (nums.some((x) => Number.isNaN(x))) return null;
  let [x1, y1, x2, y2] = nums;
  x1 = Math.max(0, Math.min(1, x1));
  y1 = Math.max(0, Math.min(1, y1));
  x2 = Math.max(0, Math.min(1, x2));
  y2 = Math.max(0, Math.min(1, y2));
  if (x2 <= x1 || y2 <= y1) return null;
  return [x1, y1, x2, y2];
}

function nodeTypeOf(data) {
  return String(data?.type || "").toLowerCase();
}

function shouldShowContent(data) {
  return !["page", "keyword"].includes(nodeTypeOf(data));
}

function shouldShowLocalPreview(data) {
  return ["table", "figure"].includes(nodeTypeOf(data));
}

function drawCropPreview(canvas, imageUrl, bbox) {
  if (!canvas || !imageUrl || !bbox) return;
  const img = new Image();
  img.onload = () => {
    const [x1, y1, x2, y2] = bbox;
    const boxW = x2 - x1;
    const boxH = y2 - y1;
    const padX = Math.max(0.012, Math.min(0.035, boxW * 0.08));
    const padY = Math.max(0.012, Math.min(0.04, boxH * 0.12));
    const bx1 = Math.max(0, x1 - padX);
    const by1 = Math.max(0, y1 - padY);
    const bx2 = Math.min(1, x2 + padX);
    const by2 = Math.min(1, y2 + padY);
    const sx = Math.max(0, Math.floor(bx1 * img.naturalWidth));
    const sy = Math.max(0, Math.floor(by1 * img.naturalHeight));
    const sw = Math.max(1, Math.min(img.naturalWidth - sx, Math.ceil((bx2 - bx1) * img.naturalWidth)));
    const sh = Math.max(1, Math.min(img.naturalHeight - sy, Math.ceil((by2 - by1) * img.naturalHeight)));

    canvas.width = sw;
    canvas.height = sh;
    canvas.style.width = `${Math.min(sw, 520)}px`;
    canvas.style.height = "auto";
    canvas.style.aspectRatio = `${sw} / ${sh}`;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.clearRect(0, 0, sw, sh);
    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
  };
  img.onerror = () => {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.width = 320;
    canvas.height = 80;
    canvas.style.width = "100%";
    canvas.style.height = "auto";
    ctx.fillStyle = "#64748b";
    ctx.font = "14px Space Grotesk";
    ctx.fillText("Crop preview unavailable", 12, 42);
  };
  img.src = imageUrl;
}

function estimateTokenCount(text) {
  const raw = String(text || "").trim();
  if (!raw) return 1;

  // Rough multilingual token estimate:
  // - CJK chars count as 1 token each
  // - Latin words count as 1 token each
  const cjkCount = (raw.match(/[\u4e00-\u9fff]/g) || []).length;
  const latinWords = (raw.replace(/[\u4e00-\u9fff]/g, " ").match(/[A-Za-z0-9_]+/g) || []).length;
  const punctuation = (raw.match(/[^\w\s\u4e00-\u9fff]/g) || []).length;

  return Math.max(1, cjkCount + latinWords + Math.floor(punctuation * 0.2));
}

function normalizeSemanticId(value) {
  return String(value || "")
    .trim()
    .toLowerCase()
    .replace(/\s+/g, "_")
    .replace(/[^\w\u4e00-\u9fff]+/g, "_")
    .replace(/^_+|_+$/g, "") || "unknown";
}

function tokenToNodeSize(tokens) {
  // Visible circle radius mapped from token count.
  const t = Math.max(1, Number(tokens) || 1);
  // Uniformly scale up all nodes while preserving the same visual format.
  const base = Math.max(14, Math.min(64, 10 + Math.sqrt(t) * 2.8));
  return Math.round(base * 3);
}

async function fetchGraph() {
  try {
    const res = await fetch(`${API_BASE}/graph`);
    const json = await res.json();
    if (!json || !json.node_map) {
      addMessage(
        "assistant",
        "Graph API is missing node details. Please restart backend server (python src/server.py)."
      );
    }
    const graph = parseGraphData(json);
    if (graph) {
      initGraph(graph);
      addMessage("assistant", "Graph loaded from backend.");
    }
  } catch (error) {
    addMessage("assistant", "Failed to load graph. Please ingest a document first.");
  }
}

async function sendChat(question) {
  const res = await fetch(`${API_BASE}/chat`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ question }),
  });
  const json = await res.json();
  if (!res.ok) {
    return json.error || "Request failed";
  }
  return json.answer;
}

function withInitialSpreadPositions(nodes) {
  // Use a deterministic spiral layout so the graph is already spread on first paint.
  // This avoids startup jitter from physics solvers on dense graphs.
  const goldenAngle = Math.PI * (3 - Math.sqrt(5)); // ~2.39996
  const baseRadius = 80;
  const radiusStep = 42;

  return nodes.map((node, index) => {
    const angle = index * goldenAngle;
    const radius = baseRadius + Math.sqrt(index + 1) * radiusStep;
    return {
      ...node,
      x: Math.cos(angle) * radius,
      y: Math.sin(angle) * radius,
    };
  });
}

function initGraph(graph) {
  graphState = graph;
  // Store global nodeMap so expandPageNode can look up content nodes
  window._graphNodeMap = graph.nodeMap || {};

  const rawNodes = graph.nodes.map((node) => {
    const color = colorMap[node.type] || colorMap.default;
    // Circle size is driven by token count.
    const tokenCount = Number(node.tokens) || estimateTokenCount(node.content || node.label);
    const size = tokenToNodeSize(tokenCount);
    return {
      id: node.id,
      label: node.label,
      // "dot" is a circular node with label rendered outside the node.
      shape: "dot",
      size,
      color: {
        background: color,
        border: color,
        highlight: { background: color, border: "#0f172a" },
        hover: { background: color, border: "#0f172a" },
      },
      borderWidth: 0,
      borderWidthSelected: 1.5,
      font: {
        color: "#0f172a",
        face: "Space Grotesk",
        size: 13,
        // Keep label outside the circle (below the node).
        vadjust: size + 8,
      },
      labelHighlightBold: false,
      shadow: false,
      selectable: true,
      data: node,
    };
  });
  const nodes = withInitialSpreadPositions(rawNodes);

  const edges = graph.edges.map((edge, index) => ({
    id: `edge_${index}`,
    from: edge.from,
    to: edge.to,
    label: edge.relation || "鍏宠仈",
    font: { color: "#cbd5f5", size: 10, align: "top" },
    arrows: { to: { enabled: true, scaleFactor: 0.7 } },
    color: { color: "rgba(100,116,139,0.55)" },
    smooth: true,
    selectable: false,
    hover: false,
    data: edge,
  }));

  nodeCount.textContent = nodes.length.toString();
  edgeCount.textContent = edges.length.toString();

  const data = { nodes: new vis.DataSet(nodes), edges: new vis.DataSet(edges) };
  const options = {
    interaction: {
      hover: true,
      multiselect: true,
      selectConnectedEdges: false,
      doubleClick: false,  // disable built-in zoom-on-dblclick, use our expand handler
    },
    // Physics disabled — 6000+ nodes with pre-computed spiral positions.
    // Enabling physics causes multi-second freezes on every interaction.
    physics: {
      enabled: false,
    },
    layout: {
      improvedLayout: false,
      randomSeed: 11,
    },
    nodes: {
      shape: "dot",
      shadow: false,
      selectable: true,
      borderWidth: 0,
      labelHighlightBold: false,
      font: { face: "Space Grotesk", size: 13, color: "#0f172a", vadjust: 12 },
    },
    edges: { smooth: true, selectable: false },
  };

  if (network) {
    network.destroy();
  }

  network = new vis.Network(graphCanvas, data, options);
  network.once("afterDrawing", () => {
    network.fit({ animation: false });
  });

  // Double-click page node to expand its content nodes
  network.on("doubleClick", (params) => {
    if (params.nodes.length === 0) return;
    const pageNode = nodes.find(n => n.id === params.nodes[0]);
    if (!pageNode || pageNode.type !== "page") return;
    const childIds = pageNode._childNodeIds;
    if (!childIds || childIds.length === 0) return;

    if (pageNode._expanded) {
      // Collapse: remove child nodes and edges
      const toRemove = childIds.map((cid, i) => cid);
      data.nodes.remove(toRemove);
      data.edges.remove(toRemove);
      pageNode._expanded = false;
      return;
    }

    // Expand: add child nodes
    const childNodes = [];
    const childEdges = [];
    const nm = window._graphNodeMap || {};
    childIds.forEach((nodeId, idx) => {
      const detail = nm[nodeId] || {};
      const nodeType = detail.type || "text";
      const snippet = detail.snippet || "";
      const figId = detail.fig_id || "";
      const label = figId || (snippet ? snippet.replace(/\s+/g, " ").slice(0, 28) + (snippet.length > 28 ? "..." : "") : nodeId.replace(/_/g, " "));
      const color = colorMap[nodeType] || colorMap.default;
      const size = tokenToNodeSize(estimateTokenCount(snippet || label));
      childNodes.push({
        id: nodeId,
        label,
        shape: "dot",
        size,
        color: { background: color, border: color, highlight: { background: color, border: "#0f172a" }, hover: { background: color, border: "#0f172a" } },
        borderWidth: 0,
        borderWidthSelected: 1.5,
        font: { color: "#0f172a", face: "Space Grotesk", size: 13, vadjust: size + 8 },
        labelHighlightBold: false,
        shadow: false,
        selectable: true,
        data: { id: nodeId, label, type: nodeType, snippet, content: snippet, source: detail.source || pageNode.source, page: detail.page, fig_id: figId, bbox: detail.bbox, image_path: detail.image_path, image_url: detail.image_url },
        x: pageNode.x + (Math.cos(idx * 2.4) * (80 + Math.random() * 60)),
        y: pageNode.y + (Math.sin(idx * 2.4) * (80 + Math.random() * 60)),
      });
      childEdges.push({ id: `expand_${nodeId}`, from: pageNode.id, to: nodeId, color: { color: "rgba(100,116,139,0.3)" }, smooth: true, selectable: false });
    });
    data.nodes.add(childNodes);
    data.edges.add(childEdges);
    pageNode._expanded = true;
  });

  network.on("click", (params) => {
    const nodeId = params.nodes?.[0];
    if (nodeId) {
      const node = nodes.find((item) => item.id === nodeId);
      if (node) {
        renderInspector(node.data);
      }
      return;
    }

    const edgeId = params.edges?.[0];
    if (edgeId) {
      const edge = edges.find((item) => item.id === edgeId);
      if (edge) {
        renderInspector(edge.data, true);
      }
    }
  });
}

function renderInspector(data, isEdge = false) {
  if (!data) return;
  nodeInspector.className = "inspector-body";

  if (isEdge) {
    nodeInspector.innerHTML = `
      <div class="inspector-card">
        <h3>Edge Relation</h3>
        <p><strong>From:</strong> ${data.from}</p>
        <p><strong>To:</strong> ${data.to}</p>
        <p><strong>Relation:</strong> ${data.relation || "鍏宠仈"}</p>
      </div>
    `;
    return;
  }

  const source = escapeHtml(data.source || "-");
  const page = data.page ?? "-";
  const nodeType = escapeHtml(data.type || "unknown");
  const keywordType = data.keyword_type
    ? `<p><strong>Keyword Type:</strong> ${escapeHtml(data.keyword_type)}</p>`
    : "";
  const nodeId = escapeHtml(data.id || "-");
  const figId = data.fig_id ? `<p><strong>Figure ID:</strong> ${escapeHtml(data.fig_id)}</p>` : "";
  const content = String(data.content || "").trim();
  const imageUrl = normalizeImageUrl(data.image_url || data.image_path);
  const bbox = normalizeBbox(data.bbox);
  const isPageNode = nodeTypeOf(data) === "page";
  const showContent = shouldShowContent(data) && content;
  const wantsLocalPreview = shouldShowLocalPreview(data);
  const showLocalPreview = wantsLocalPreview && bbox && imageUrl;
  const showPagePreview = isPageNode && imageUrl;
  const previewUnavailable = wantsLocalPreview && !showLocalPreview
    ? `<p class="inspector-note">${
        bbox
          ? "Local Preview unavailable: page image is missing."
          : "Local Preview unavailable: this node has no bbox coordinates."
      }</p>`
    : "";
  const cropCanvasId = `crop_${String(data.id || "node").replace(/[^a-zA-Z0-9_]/g, "_")}`;

  nodeInspector.innerHTML = `
    <div class="inspector-card">
      <h3>${escapeHtml(data.label || data.id || "Node")}</h3>
      <p><strong>Type:</strong> ${nodeType}</p>
      ${keywordType}
      <p><strong>Source:</strong> ${source}</p>
      <p><strong>Page:</strong> ${page}</p>
      ${figId}
      <p><strong>ID:</strong> ${nodeId}</p>
      ${
        showContent
          ? `<p><strong>Content:</strong></p><pre class="inspector-content">${escapeHtml(content)}</pre>`
          : ""
      }
      ${
        showPagePreview
          ? `<p><strong>Page Preview:</strong></p><img src="${escapeHtml(imageUrl)}" alt="page preview" class="page-preview-image" />`
          : ""
      }
      ${
        showLocalPreview
          ? `<p><strong>Local Preview:</strong></p><div class="crop-preview-wrap"><canvas id="${cropCanvasId}" class="crop-preview-canvas"></canvas></div>`
          : ""
      }
      ${previewUnavailable}
    </div>
  `;

  if (showLocalPreview) {
    const canvas = document.getElementById(cropCanvasId);
    drawCropPreview(canvas, imageUrl, bbox);
  }
}

function parseGraphData(raw) {
  if (!raw) return null;

  const nodeMap = raw.node_map || {};
  const documents = raw.documents || [{ source: raw.source || "doc", pages: raw.pages || [] }];
  const nodes = [];
  const edges = [];

  documents.forEach((doc) => {
    const source = doc.source || "doc";
    const pages = doc.pages || [];

    pages.forEach((page) => {
      const pageLabel = `${source} p${page.page}`;
      const pageId = `${source}_page_${page.page}`;
      const childNodeIds = (page.node_ids || []).slice();

      nodes.push({
        id: pageId,
        label: pageLabel,
        type: "page",
        tokens: Math.max(16, childNodeIds.length * 3),
        source,
        page: page.page,
        content: "",
        image_path: page.image_path || "",
        image_url: page.image_url || "",
        _childNodeIds: childNodeIds,
        _expanded: false,
      });

      (page.keywords || []).forEach((keyword) => {
        const term =
          typeof keyword === "string"
            ? keyword
            : String(keyword?.term || keyword?.name || "").trim();
        if (!term) return;
        const keywordType =
          typeof keyword === "string" ? "concept" : String(keyword?.type || "concept");
        const keywordId = `keyword_${normalizeSemanticId(term)}`;
        nodes.push({
          id: keywordId,
          label: term,
          type: "keyword",
          tokens: estimateTokenCount(term),
          keyword_type: keywordType,
          source,
          page: page.page,
          content: term,
          image_path: page.image_path || "",
          image_url: page.image_url || "",
        });
        edges.push({ from: pageId, to: keywordId, relation: "keyword" });
      });

      (page.relations || []).forEach((rel) => {
        const fromRaw = String(rel?.from || "").trim();
        const toRaw = String(rel?.to || "").trim();
        if (!fromRaw || !toRaw) return;
        const fromId = `keyword_${normalizeSemanticId(fromRaw)}`;
        const toId = `keyword_${normalizeSemanticId(toRaw)}`;
        nodes.push({
          id: fromId,
          label: fromRaw,
          type: "keyword",
          tokens: estimateTokenCount(fromRaw),
          keyword_type: "concept",
          source,
          page: page.page,
          content: fromRaw,
          image_path: page.image_path || "",
          image_url: page.image_url || "",
        });
        nodes.push({
          id: toId,
          label: toRaw,
          type: "keyword",
          tokens: estimateTokenCount(toRaw),
          keyword_type: "concept",
          source,
          page: page.page,
          content: toRaw,
          image_path: page.image_path || "",
          image_url: page.image_url || "",
        });
        edges.push({ from: fromId, to: toId, relation: rel.relation });
      });
    });
  });

  const uniqueNodes = Array.from(new Map(nodes.map((n) => [n.id, n])).values());
  return { nodes: uniqueNodes, edges, nodeMap: raw.node_map || {} };
}

sendBtn.addEventListener("click", () => {
  const text = chatInput.value.trim();
  if (!text) return;
  addMessage("user", text);
  chatInput.value = "";
  sendChat(text)
    .then((answer) => addMessage("assistant", answer))
    .catch(() => addMessage("assistant", "Chat request failed."));
});

chatInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter" && !event.shiftKey) {
    event.preventDefault();
    sendBtn.click();
  }
});

fileInput.addEventListener("change", async (event) => {
  const file = event.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append("file", file);

  addMessage("assistant", "Ingesting document... this may take a moment.");

  try {
    const res = await fetch(`${API_BASE}/ingest`, { method: "POST", body: formData });
    const json = await res.json();
    if (!res.ok) {
      addMessage("assistant", json.error || "Ingest failed.");
      return;
    }
    addMessage("assistant", "Ingest completed. Reloading graph...");
    await fetchGraph();
  } catch (error) {
    addMessage("assistant", "Ingest request failed.");
  }
});

resetGraph.addEventListener("click", () => {
  initGraph(demoGraph);
  nodeInspector.className = "inspector-empty";
  nodeInspector.textContent = "Select a node to see details.";
});

centerGraph.addEventListener("click", () => {
  if (network) {
    network.fit({ animation: { duration: 600, easingFunction: "easeInOutQuad" } });
  }
});

initGraph(demoGraph);
addMessage("assistant", "Ready. Upload a document to ingest and visualize the graph.");
fetchGraph();
