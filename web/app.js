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
  entity: "#60a5fa",
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
    { id: "entity_qwen", label: "Qwen-VL", type: "entity", tokens: 40 },
    { id: "entity_acc", label: "Accuracy", type: "entity", tokens: 30 },
  ],
  edges: [
    { from: "doc_text_1", to: "doc_table_1", relation: "同页" },
    { from: "doc_table_1", to: "doc_fig_1", relation: "支撑结论" },
    { from: "doc_fig_1", to: "entity_acc", relation: "展示" },
    { from: "entity_qwen", to: "doc_fig_1", relation: "表现最优" },
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
    label: edge.relation || "关联",
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
        <p><strong>Relation:</strong> ${data.relation || "关联"}</p>
      </div>
    `;
    return;
  }

  const source = escapeHtml(data.source || "-");
  const page = data.page ?? "-";
  const nodeType = escapeHtml(data.type || "unknown");
  const nodeId = escapeHtml(data.id || "-");
  const figId = data.fig_id ? `<p><strong>Figure ID:</strong> ${escapeHtml(data.fig_id)}</p>` : "";
  const content = String(data.content || "").trim();
  const imageUrl = normalizeImageUrl(data.image_url || data.image_path);

  nodeInspector.innerHTML = `
    <div class="inspector-card">
      <h3>${escapeHtml(data.label || data.id || "Node")}</h3>
      <p><strong>Type:</strong> ${nodeType}</p>
      <p><strong>Source:</strong> ${source}</p>
      <p><strong>Page:</strong> ${page}</p>
      ${figId}
      <p><strong>ID:</strong> ${nodeId}</p>
      ${
        content
          ? `<p><strong>Content:</strong></p><pre style="white-space: pre-wrap; word-break: break-word; max-height: 220px; overflow-y: auto; padding: 10px; border-radius: 10px; background: rgba(15,23,42,0.05); border: 1px solid rgba(15,23,42,0.08);">${escapeHtml(content)}</pre>`
          : `<p><strong>Content:</strong> (暂无原文内容)</p>`
      }
      ${
        imageUrl
          ? `<p><strong>Page Preview:</strong></p><img src="${escapeHtml(imageUrl)}" alt="page preview" style="width: 100%; border-radius: 10px; border: 1px solid rgba(15,23,42,0.12);" />`
          : ""
      }
    </div>
  `;
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
          content,
          image_path: detail.image_path || page.image_path || "",
          image_url: detail.image_url || page.image_url || "",
        });
        edges.push({ from: pageId, to: nodeId, relation: "contains" });
      });

      (page.entities || []).forEach((entity) => {
        const entityId = `entity_${entity.name}`.replace(/\s+/g, "_");
        nodes.push({
          id: entityId,
          label: entity.name,
          type: "entity",
          tokens: estimateTokenCount(entity.name),
          source,
          page: page.page,
          content: "",
          image_path: page.image_path || "",
          image_url: page.image_url || "",
        });
        edges.push({ from: pageId, to: entityId, relation: "mentions" });
      });

      (page.relations || []).forEach((rel) => {
        const fromId = `entity_${rel.from}`.replace(/\s+/g, "_");
        const toId = `entity_${rel.to}`.replace(/\s+/g, "_");
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
