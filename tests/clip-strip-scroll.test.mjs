import assert from "node:assert/strict";
import { readFileSync } from "node:fs";
import test from "node:test";
import vm from "node:vm";

const source = readFileSync(
    process.env.H3_EXTENDER_SOURCE || new URL("../web/extender.js", import.meta.url),
    "utf8",
).replace(/^import .*;\r?\n/gm, "");

// A small DOM fixture for the real render() path. Emptying a scroller clamps
// its position, as a browser layout read does while the cards are rebuilt.
// Smooth scrolls remain pending until flushFrames(), so an animated restoration
// cannot pass the immediate viewport assertions below.
class Element {
    constructor(tagName, document) {
        this.tagName = tagName.toUpperCase();
        this.document = document;
        this.children = [];
        this.dataset = {};
        this.style = { setProperty() {} };
        this.listeners = new Map();
        this.clientWidth = 740;
        this._scrollLeft = 0;
        this.scrollTop = 0;
    }

    append(...children) {
        for (const child of children) {
            child.parentElement = this;
            this.children.push(child);
        }
    }

    appendChild(child) {
        this.append(child);
        return child;
    }

    replaceChildren(...children) {
        this.children = [];
        this._scrollLeft = 0;
        this.scrollTop = 0;
        this.append(...children);
    }

    get scrollWidth() {
        const widths = this.children.map((child) =>
            parseFloat(child.style?.width || child.style?.minWidth) || 0
        );
        return widths.reduce((sum, width) => sum + width, 0)
            + Math.max(0, widths.length - 1) * (parseFloat(this.style.gap) || 0);
    }

    get scrollLeft() {
        return this._scrollLeft;
    }

    set scrollLeft(value) {
        this.scrollTo({ left: value });
    }

    scrollTo({ left = this._scrollLeft, top = this.scrollTop, behavior = "auto" }) {
        const apply = () => {
            this._scrollLeft = Math.max(0, Math.min(left, this.scrollWidth - this.clientWidth));
            this.scrollTop = Math.max(0, top);
        };
        if (behavior === "smooth" || (behavior === "auto" && this.style.scrollBehavior === "smooth")) {
            this.document.frames.push(apply);
        } else {
            apply();
        }
    }

    addEventListener(type, handler) {
        this.listeners.set(type, handler);
    }

    querySelectorAll(selector) {
        const descendants = this.children.flatMap((child) => [child, ...child.querySelectorAll(selector)]);
        return descendants.filter((child) =>
            selector === "textarea[data-h3-prompt-clip-id]"
                && child.tagName === "TEXTAREA" && child.dataset.h3PromptClipId
        );
    }

    closest() { return null; }
    setSelectionRange() {}
}

function makeFixture(mode = "ref2va", nodes2 = false) {
    const document = {
        frames: [],
        activeElement: null,
        getElementById() { return {}; },
        createElement(tagName) { return new Element(tagName, this); },
        createTextNode(text) {
            const node = new Element("#text", this);
            node.textContent = text;
            return node;
        },
    };
    const context = vm.createContext({
        app: { registerExtension() {}, graph: null },
        api: { apiURL(value) { return value; } },
        LiteGraph: { vueNodesMode: nodes2 },
        document,
        requestAnimationFrame(callback) { document.frames.push(callback); },
        alert() {},
        console,
        Set,
        Map,
    });
    vm.runInContext(source, context);
    context.mode = mode;
    vm.runInContext(`
        node = { id: 42, widgets: [], inputs: [], properties: {} };
        runtime = {
            root: document.createElement("div"),
            cards: document.createElement("div"),
            refsRow: document.createElement("div"),
            counter: document.createElement("span"),
            status: document.createElement("span"),
            state: {
                generation_mode: mode,
                motion_context: true,
                clips: Array.from({ length: 20 }, (_, i) => newClip(i)),
            },
            refsState: { refs: [] },
            jsonWidget: { value: "" },
            loraNames: [],
            cachedClipIds: new Set(),
            validatedClipIds: new Set(),
            computedClipIds: new Set(),
            computedIndices: new Set(),
            validatedCount: 0,
            activeClipIndex: -1,
            activePhase: "idle",
        };
        runtime.cards.style.gap = "9px";
        runtime.cards.style.scrollBehavior = "smooth";
        render(node, runtime);
    `, context);
    const flushFrames = () => {
        while (document.frames.length) document.frames.shift()();
    };
    flushFrames();
    return {
        context,
        runtime: context.runtime,
        render: () => vm.runInContext("render(node, runtime)", context),
        flushFrames,
    };
}

for (const nodes2 of [false, true]) {
    test(`${nodes2 ? "Nodes 2.0" : "Legacy"}: motion redraw keeps the clip-strip viewport immediately`, () => {
        const { runtime, render, flushFrames } = makeFixture("ref2va", nodes2);
        runtime.cards.scrollTo({ left: 4000, top: 30, behavior: "instant" });
        runtime.refsRow.scrollTo({ left: 120, behavior: "instant" });
        runtime.state.motion_context = false;

        render();

        assert.equal(runtime.cards.scrollLeft, 4000);
        assert.equal(runtime.cards.scrollTop, 30);
        assert.equal(runtime.refsRow.scrollLeft, 120);
        flushFrames();
        assert.equal(runtime.cards.scrollLeft, 4000);
    });
}

test("FL2VA first/last frames stay aligned with the restored clip strip", () => {
    const { runtime, render, flushFrames } = makeFixture("fl2va");
    runtime.cards.scrollTo({ left: 4000, behavior: "instant" });

    render();

    assert.equal(runtime.cards.scrollLeft, 4000);
    assert.equal(runtime.refsRow.scrollLeft, 4000);
    flushFrames();
    assert.equal(runtime.refsRow.scrollLeft, runtime.cards.scrollLeft);
});

test("removing clips clamps the viewport to the shortened timeline", () => {
    const { runtime, render } = makeFixture("fl2va");
    runtime.cards.scrollTo({ left: 4000, behavior: "instant" });
    runtime.state.clips.splice(5);

    render();

    const end = runtime.cards.scrollWidth - runtime.cards.clientWidth;
    assert.equal(runtime.cards.scrollLeft, end);
    assert.equal(runtime.refsRow.scrollLeft, end);
    runtime.state.clips.splice(1);
    render();
    assert.equal(runtime.cards.scrollLeft, 0);
    assert.equal(runtime.refsRow.scrollLeft, 0);
});

test("deferred layout work does not overwrite later user or Add Clip navigation", () => {
    const { runtime, render, flushFrames } = makeFixture("ref2va", true);
    runtime.cards.scrollTo({ left: 4000, behavior: "instant" });
    render();
    runtime.cards.scrollTo({ left: 1800, behavior: "instant" });
    flushFrames();
    assert.equal(runtime.cards.scrollLeft, 1800);

    render();
    // Add Clip intentionally scrolls to the end after rendering. Preserve
    // that navigation instead of replaying a stale viewport in a later frame.
    runtime.cards.scrollLeft = runtime.cards.scrollWidth;
    flushFrames();
    assert.equal(runtime.cards.scrollLeft, runtime.cards.scrollWidth - runtime.cards.clientWidth);
});
