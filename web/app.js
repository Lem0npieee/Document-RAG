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
  bubble.textContent = text;
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

function drawCropPreview(canvas, imageUrl, bbox) {
  if (!canvas || !imageUrl || !bbox) return;
  const img = new Image();
  img.onload = () => {
    const [x1, y1, x2, y2] = bbox;
    // Keep crop accurate by default; only add tiny padding for very small boxes.
    const boxW = x2 - x1;
    const boxH = y2 - y1;
    const tinyPad = boxW < 0.08 || boxH < 0.05 ? 0.005 : 0;
    const padX = tinyPad;
    const padY = tinyPad;
    const bx1 = Math.max(0, x1 - padX);
    const by1 = Math.max(0, y1 - padY);
    const bx2 = Math.min(1, x2 + padX);
    const by2 = Math.min(1, y2 + padY);

    let sx = Math.max(0, Math.floor(bx1 * img.width));
    let sy = Math.max(0, Math.floor(by1 * img.height));
    let sw = Math.max(1, Math.floor((bx2 - bx1) * img.width));
    let sh = Math.max(1, Math.floor((by2 - by1) * img.height));

    // Refine crop by trimming near-white margins inside the bbox region.
    const probe = document.createElement("canvas");
    probe.width = sw;
    probe.height = sh;
    const pctx = probe.getContext("2d", { willReadFrequently: true });
    if (pctx) {
      pctx.drawImage(img, sx, sy, sw, sh, 0, 0, sw, sh);
      const imageData = pctx.getImageData(0, 0, sw, sh).data;
      let minX = sw;
      let minY = sh;
      let maxX = -1;
      let maxY = -1;

      const stepX = Math.max(1, Math.floor(sw / 320));
      const stepY = Math.max(1, Math.floor(sh / 320));
      const isForeground = (r, g, b) => {
        const brightness = (r + g + b) / 3;
        const spread = Math.max(r, g, b) - Math.min(r, g, b);
        return brightness < 246 || spread > 12;
      };

      for (let y = 0; y < sh; y += stepY) {
        for (let x = 0; x < sw; x += stepX) {
          const idx = (y * sw + x) * 4;
          const r = imageData[idx];
          const g = imageData[idx + 1];
          const b = imageData[idx + 2];
          if (!isForeground(r, g, b)) continue;
          if (x < minX) minX = x;
          if (y < minY) minY = y;
          if (x > maxX) maxX = x;
          if (y > maxY) maxY = y;
        }
      }

      if (maxX >= minX && maxY >= minY) {
        const refineMargin = 4;
        const rsx = Math.max(0, minX - refineMargin);
        const rsy = Math.max(0, minY - refineMargin);
        const rsw = Math.max(1, Math.min(sw - rsx, maxX - minX + 1 + refineMargin * 2));
        const rsh = Math.max(1, Math.min(sh - rsy, maxY - minY + 1 + refineMargin * 2));
        sx += rsx;
        sy += rsy;
        sw = rsw;
        sh = rsh;
      }
    }

    // Keep display reasonably large while avoiding extreme upscale blur.
    const maxDisplayW = 560;
    const minDisplayW = 220;
    const displayW = Math.max(minDisplayW, Math.min(maxDisplayW, sw));
    const displayH = Math.max(120, Math.round((displayW * sh) / sw));
    const dpr = Math.max(1, window.devicePixelRatio || 1);

    canvas.style.width = `${displayW}px`;
    canvas.style.height = `${displayH}px`;
    canvas.width = Math.round(displayW * dpr);
    canvas.height = Math.round(displayH * dpr);
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.imageSmoothingEnabled = true;
    ctx.imageSmoothingQuality = "high";
    ctx.clearRect(0, 0, displayW, displayH);
    ctx.drawImage(img, sx, sy, sw, sh, 0, 0, displayW, displayH);
  };
  img.onerror = () => {
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    canvas.style.width = "100%";
    canvas.style.height = "80px";
    canvas.width = 320;
    canvas.height = 80;
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
      // Prevent selecting connected edges when clicking a node.
      selectConnectedEdges: false,
    },
    // Re-enable physics and increase repulsion so nodes spread further apart.
    physics: {
      enabled: true,
      solver: "forceAtlas2Based",
      stabilization: {
        enabled: true,
        iterations: 300,
        updateInterval: 20,
        fit: true,
      },
      forceAtlas2Based: {
        gravitationalConstant: -200,
        centralGravity: 0.002,
        // Stronger edge attraction.
        springLength: 125,
        springConstant: 0.2,
        damping: 0.85,
        avoidOverlap: 0.8,
      },
      minVelocity: 0.2,
      maxVelocity: 30,
      timestep: 0.35,
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
    network.fit({ animation: { duration: 900, easingFunction: "easeInOutQuad" } });
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
  const isPageNode = String(data.type || "").toLowerCase() === "page";
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
        content
          ? `<p><strong>Content:</strong></p><pre style="white-space: pre-wrap; word-break: break-word; max-height: 220px; overflow-y: auto; padding: 10px; border-radius: 10px; background: rgba(15,23,42,0.05); border: 1px solid rgba(15,23,42,0.08);">${escapeHtml(content)}</pre>`
          : `<p><strong>Content:</strong> (鏆傛棤鍘熸枃鍐呭)</p>`
      }
      ${
        isPageNode
          ? imageUrl
            ? `<p><strong>Page Preview:</strong></p><img src="${escapeHtml(imageUrl)}" alt="page preview" style="width: 100%; border-radius: 10px; border: 1px solid rgba(15,23,42,0.12);" />`
            : `<p class="inspector-note">No page image available.</p>`
          : imageUrl
            ? `${
                bbox
                  ? `<p><strong>Local Preview:</strong></p><canvas id="${cropCanvasId}" class="crop-preview-canvas"></canvas>`
                  : `<p class="inspector-note">No local bbox for this node yet.</p>`
              }`
            : `<p class="inspector-note">No image available for this node.</p>`
      }
    </div>
  `;

  if (!isPageNode && bbox && imageUrl) {
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
      nodes.push({
        id: pageId,
        label: pageLabel,
        type: "page",
        tokens: Math.max(16, (page.node_ids || []).length * 3),
        source,
        page: page.page,
        content: "",
        image_path: page.image_path || "",
        image_url: page.image_url || "",
      });

      (page.node_ids || []).forEach((nodeId) => {
        const detail = nodeMap[nodeId] || {};
        const nodeType = detail.type || "text";
        const content = detail.content || "";
        const figId = detail.fig_id || "";
        const label =
          figId ||
          (content
            ? content.replace(/\s+/g, " ").slice(0, 28) + (content.length > 28 ? "..." : "")
            : nodeId.replace(/_/g, " "));
        nodes.push({
          id: nodeId,
          label,
          type: nodeType,
          tokens: estimateTokenCount(content || label),
          source: detail.source || source,
          page: detail.page || page.page,
          fig_id: figId,
          bbox: detail.bbox || null,
          content,
          image_path: detail.image_path || page.image_path || "",
          image_url: detail.image_url || page.image_url || "",
        });
        edges.push({ from: pageId, to: nodeId, relation: "contains" });
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
  return { nodes: uniqueNodes, edges };
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
