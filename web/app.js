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

async function fetchGraph() {
  try {
    const res = await fetch(`${API_BASE}/graph`);
    const json = await res.json();
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

function initGraph(graph) {
  graphState = graph;

  const nodes = graph.nodes.map((node) => {
    const color = colorMap[node.type] || colorMap.default;
    const size = 10 + Math.min(node.tokens || 40, 140) * 0.12;
    return {
      id: node.id,
      label: node.label,
      color,
      value: size,
      font: { color: "#0f172a", face: "Space Grotesk" },
      shadow: false,
    selectable: true,
      data: node,
    };
  });

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
    interaction: { hover: true, multiselect: true },
    physics: {
      stabilization: { iterations: 120, fit: true },
      barnesHut: {
        gravitationalConstant: -24000,
        springLength: 160,
        springConstant: 0.04,
        damping: 0.3,
      },
    },
    layout: { improvedLayout: true },
    nodes: { shadow: false, selectable: true },
    edges: { smooth: true, selectable: false },
  };

  if (network) {
    network.destroy();
  }

  network = new vis.Network(graphCanvas, data, options);
  network.once("afterDrawing", () => {
    network.fit({ animation: { duration: 900, easingFunction: "easeInOutQuad" } });
  });

  network.on("selectNode", (params) => {
    const nodeId = params.nodes[0];
    const node = nodes.find((item) => item.id === nodeId);
    if (node) {
      renderInspector(node.data);
    }
  });

  network.on("selectEdge", (params) => {
    const edgeId = params.edges[0];
    const edge = edges.find((item) => item.id === edgeId);
    if (edge) {
      renderInspector(edge.data, true);
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

  nodeInspector.innerHTML = `
    <div class="inspector-card">
      <h3>${data.label}</h3>
      <p><strong>Type:</strong> ${data.type}</p>
      <p><strong>Token size:</strong> ${data.tokens || 0}</p>
      <p><strong>ID:</strong> ${data.id}</p>
    </div>
  `;
}

function parseGraphData(raw) {
  if (!raw) return null;

  const documents = raw.documents || [{ source: raw.source || "doc", pages: raw.pages || [] }];
  const nodes = [];
  const edges = [];

  documents.forEach((doc) => {
    const source = doc.source || "doc";
    const pages = doc.pages || [];

    pages.forEach((page) => {
      const pageLabel = `${source} p${page.page}`;
      const pageId = `${source}_page_${page.page}`;
      nodes.push({ id: pageId, label: pageLabel, type: "text", tokens: 120 });

      (page.node_ids || []).forEach((nodeId) => {
        nodes.push({
          id: nodeId,
          label: nodeId.replace(/_/g, " "),
          type: "text",
          tokens: 60,
        });
        edges.push({ from: pageId, to: nodeId, relation: "contains" });
      });

      (page.entities || []).forEach((entity) => {
        const entityId = `entity_${entity.name}`.replace(/\s+/g, "_");
        nodes.push({
          id: entityId,
          label: entity.name,
          type: "entity",
          tokens: 40,
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
