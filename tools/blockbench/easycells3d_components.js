(() => {
    const PLUGIN_ID = 'easycells3d_components';
    const SETTINGS_KEY = 'easycells3d.blockbench.settings';
    const CACHE_KEY = 'easycells3d.blockbench.components';
    const DEFAULT_SETTINGS = {
        project_root: '',
        python_command: 'python',
        main_script: 'main.py',
        export_path: 'Assets/Blockbench/scene.ec3d.json',
        glb_export_path: 'Assets/Blockbench/scene.glb',
        unit_scale: 16,
        run_args: ''
    };

    let actions = [];
    let easycellsIdCubeProperty;
    let easycellsIdGroupProperty;
    let componentsCubeProperty;
    let componentsGroupProperty;

    function settings() {
        try {
            return Object.assign({}, DEFAULT_SETTINGS, JSON.parse(localStorage.getItem(SETTINGS_KEY) || '{}'));
        } catch (error) {
            return Object.assign({}, DEFAULT_SETTINGS);
        }
    }

    function saveSettings(next) {
        localStorage.setItem(SETTINGS_KEY, JSON.stringify(Object.assign({}, settings(), next)));
    }

    function componentCache() {
        try {
            return JSON.parse(localStorage.getItem(CACHE_KEY) || '{"components": []}');
        } catch (error) {
            return {components: []};
        }
    }

    function saveComponentCache(cache) {
        localStorage.setItem(CACHE_KEY, JSON.stringify(cache || {components: []}));
    }

    function selectedElement() {
        if (typeof Cube !== 'undefined' && Cube.selected && Cube.selected.length) {
            return Cube.selected[0];
        }
        if (typeof Group !== 'undefined' && Group.selected) {
            if (Array.isArray(Group.selected) && Group.selected.length) return Group.selected[0];
            if (!Array.isArray(Group.selected)) return Group.selected;
        }
        if (typeof Outliner !== 'undefined' && Outliner.selected && Outliner.selected.length) {
            return Outliner.selected.find(isSupportedElement) || Outliner.selected[0];
        }
        return null;
    }

    function isSupportedElement(element) {
        if (!element) return false;
        if (typeof Cube !== 'undefined' && element instanceof Cube) return true;
        if (typeof Group !== 'undefined' && element instanceof Group) return true;
        if (element.type === 'cube' || element.type === 'group') return true;
        if (element.constructor && ['Cube', 'Group'].includes(element.constructor.name)) return true;
        if (element.is_group === true) return true;
        if (Array.isArray(element.from) && Array.isArray(element.to)) return true;
        if (Array.isArray(element.children) && !Array.isArray(element.from)) return true;
        return false;
    }

    function selectionHelpMessage() {
        const selected = typeof Outliner !== 'undefined' && Outliner.selected
            ? Outliner.selected.map(element => `${element.name || element.uuid || '<unnamed>'} (${element.type || element.constructor?.name || 'unknown'})`).join(', ')
            : 'nothing';
        return `Select a cube or group in the Outliner while in Edit mode. Current selection: ${selected}`;
    }

    function ensureId(element) {
        if (!element.easycells_id) {
            element.easycells_id = createId();
            if (typeof element.updateElement === 'function') element.updateElement();
        }
        return element.easycells_id;
    }

    function componentsOf(element) {
        try {
            const raw = element.easycells_components || '[]';
            const value = typeof raw === 'string' ? JSON.parse(raw) : raw;
            if (Array.isArray(value)) return value.map(migrateComponent);
            if (value && Array.isArray(value.components)) return value.components.map(migrateComponent);
        } catch (error) {
        }
        return [];
    }

    function saveComponents(element, components) {
        element.easycells_components = JSON.stringify(components || []);
        if (typeof element.updateElement === 'function') element.updateElement();
    }

    function migrateComponent(component) {
        const next = Object.assign({}, component);
        if (next.config && !next.args && !next.fields) {
            next.args = {};
            next.fields = next.config;
            delete next.config;
        }
        next.args = next.args || {};
        next.fields = next.fields || {};
        return next;
    }

    function componentMetadata(type) {
        return componentCache().components.find(component => component.name === type || component.class_path === type);
    }

    function componentOptions() {
        const components = componentCache().components;
        const options = {};
        components.forEach(component => {
            options[component.name] = component.name;
        });
        return options;
    }

    function configureDialog() {
        const current = settings();
        new Dialog({
            id: 'easycells3d_blockbench_config',
            title: 'EasyCells3D Settings',
            form: {
                project_root: {label: 'Project Root', type: 'text', value: current.project_root},
                python_command: {label: 'Python Command', type: 'text', value: current.python_command},
                main_script: {label: 'Main Script', type: 'text', value: current.main_script},
                export_path: {label: 'Scene Export Path', type: 'text', value: current.export_path},
                glb_export_path: {label: 'Visual GLB Export Path', type: 'text', value: current.glb_export_path},
                unit_scale: {label: 'Blockbench Units Per EasyCells Unit', type: 'number', value: current.unit_scale},
                run_args: {label: 'Run Arguments', type: 'text', value: current.run_args}
            },
            onConfirm(form) {
                saveSettings(form);
                Blockbench.showQuickMessage('EasyCells3D settings saved');
            }
        }).show();
    }

    function refreshComponents() {
        const current = settings();
        if (!current.project_root) {
            Blockbench.showMessageBox({title: 'EasyCells3D', message: 'Configure Project Root first.'});
            return;
        }
        if (!isDesktop()) {
            Blockbench.showMessageBox({title: 'EasyCells3D', message: 'Component refresh requires the desktop app.'});
            return;
        }

        try {
            const childProcess = require('child_process');
            const path = require('path');
            const script = path.join(current.project_root, 'tools', 'blockbench', 'discover_components.py');
            const python = resolvePythonCommand(current.python_command, current.project_root);
            const output = childProcess.execFileSync(python, [script, current.project_root], {
                cwd: current.project_root,
                encoding: 'utf8',
                windowsHide: true
            });
            const cache = JSON.parse(output);
            saveComponentCache(cache);
            Blockbench.showQuickMessage(`${cache.components.length} EasyCells3D components found`);
        } catch (error) {
            Blockbench.showMessageBox({
                title: 'EasyCells3D Refresh Failed',
                message: String(error.stderr || error.message || error)
            });
        }
    }

    function addComponentDialog() {
        const element = selectedElement();
        if (!isSupportedElement(element)) {
            Blockbench.showQuickMessage(selectionHelpMessage());
            return;
        }
        const options = componentOptions();
        if (!Object.keys(options).length) {
            Blockbench.showQuickMessage('Refresh components first');
            return;
        }

        new Dialog({
            id: 'easycells3d_add_component',
            title: 'Add EasyCells3D Component',
            form: {
                type: {label: 'Component', type: 'select', options}
            },
            onConfirm(form) {
                ensureId(element);
                const metadata = componentMetadata(form.type) || {};
                const component = {
                    type: form.type,
                    args: defaultsForParameters([].concat(metadata.required_args || [], metadata.optional_args || [])),
                    fields: defaultsForFields(metadata.fields || {})
                };
                const components = componentsOf(element);
                components.push(component);
                saveComponents(element, components);
                Blockbench.showQuickMessage(`Added ${form.type}`);
            }
        }).show();
    }

    function editComponentsDialog() {
        const element = selectedElement();
        if (!isSupportedElement(element)) {
            Blockbench.showQuickMessage(selectionHelpMessage());
            return;
        }
        ensureId(element);
        const components = componentsOf(element);
        if (!components.length) {
            Blockbench.showQuickMessage('No EasyCells3D components on selection');
            return;
        }

        const componentChoices = {};
        components.forEach((component, index) => {
            componentChoices[String(index)] = `${index + 1}. ${component.type}`;
        });

        new Dialog({
            id: 'easycells3d_pick_component',
            title: 'Edit EasyCells3D Component',
            form: {
                index: {label: 'Component', type: 'select', options: componentChoices}
            },
            onConfirm(form) {
                editSingleComponentDialog(element, Number(form.index));
            }
        }).show();
    }

    function editSingleComponentDialog(element, index) {
        const components = componentsOf(element);
        const component = components[index];
        if (!component) return;
        const metadata = componentMetadata(component.type) || {};
        const form = {};

        [].concat(metadata.required_args || [], metadata.optional_args || []).forEach(param => {
            form[`arg:${param.name}`] = dialogField(param, component.args[param.name]);
        });
        Object.keys(metadata.fields || {}).forEach(name => {
            form[`field:${name}`] = dialogField(metadata.fields[name], component.fields[name]);
        });

        new Dialog({
            id: 'easycells3d_edit_component',
            title: component.type,
            form,
            onConfirm(values) {
                Object.keys(values).forEach(key => {
                    const [section, name] = key.split(':');
                    if (section === 'arg') component.args[name] = coerceValue(values[key], findParam(metadata, name));
                    if (section === 'field') component.fields[name] = exportFieldValue(values[key], (metadata.fields || {})[name]);
                });
                components[index] = component;
                saveComponents(element, components);
                Blockbench.showQuickMessage(`${component.type} updated`);
            }
        }).show();
    }

    function removeComponentDialog() {
        const element = selectedElement();
        if (!isSupportedElement(element)) return;
        const components = componentsOf(element);
        if (!components.length) {
            Blockbench.showQuickMessage('No EasyCells3D components on selection');
            return;
        }

        const options = {};
        components.forEach((component, index) => {
            options[String(index)] = `${index + 1}. ${component.type}`;
        });
        new Dialog({
            id: 'easycells3d_remove_component',
            title: 'Remove EasyCells3D Component',
            form: {
                index: {label: 'Component', type: 'select', options}
            },
            onConfirm(form) {
                components.splice(Number(form.index), 1);
                saveComponents(element, components);
                Blockbench.showQuickMessage('Component removed');
            }
        }).show();
    }

    function exportScene(runAfterExport) {
        const current = settings();
        const visualPath = resolveProjectPath(current.glb_export_path, current.project_root);
        const scene = buildScene(visualPath);
        const content = JSON.stringify(scene, null, 2);

        if (isDesktop() && current.export_path && current.project_root) {
            const path = require('path');
            const exportPath = resolveProjectPath(current.export_path, current.project_root);
            exportVisualGlb(visualPath, glbResult => {
                Blockbench.writeFile(exportPath, {content}, () => {
                    const suffix = glbResult.ok ? ` + ${path.basename(visualPath)}` : ' (scene only)';
                    Blockbench.showQuickMessage(`Exported ${path.basename(exportPath)}${suffix}`);
                    if (!glbResult.ok) {
                        Blockbench.showMessageBox({
                            title: 'EasyCells3D GLB Export',
                            message: glbResult.message
                        });
                    }
                    if (runAfterExport) runProject();
                });
            });
            return;
        }

        Blockbench.export({
            type: 'EasyCells3D Scene',
            extensions: ['json'],
            name: 'scene.ec3d',
            content
        }, () => {
            if (runAfterExport) runProject();
        });
    }

    function runProject() {
        const current = settings();
        if (!isDesktop()) {
            Blockbench.showQuickMessage('Run requires the desktop app');
            return;
        }
        try {
            const childProcess = require('child_process');
            const path = require('path');
            const python = resolvePythonCommand(current.python_command, current.project_root);
            const script = path.isAbsolute(current.main_script)
                ? current.main_script
                : path.join(current.project_root, current.main_script);
            const args = [script].concat(splitArgs(current.run_args));
            childProcess.spawn(python, args, {
                cwd: current.project_root,
                detached: true,
                windowsHide: false,
                stdio: 'ignore'
            }).unref();
            Blockbench.showQuickMessage(`Launched ${current.main_script}`);
        } catch (error) {
            Blockbench.showMessageBox({title: 'EasyCells3D Run Failed', message: String(error.message || error)});
        }
    }

    function exportVisualGlb(filePath, callback) {
        if (!isDesktop()) {
            callback({ok: false, message: 'Automatic GLB export requires the desktop app.'});
            return;
        }
        if (!filePath) {
            callback({ok: false, message: 'Configure Visual GLB Export Path first.'});
            return;
        }

        const fs = require('fs');
        const path = require('path');
        fs.mkdirSync(path.dirname(filePath), {recursive: true});

        try {
            const codec = gltfCodec();
            if (!codec || typeof codec.compile !== 'function') {
                callback({
                    ok: false,
                    message: 'Could not find a glTF/GLB codec in this Blockbench version. Export the visual model manually as GLB to the configured path.'
                });
                return;
            }

            const compiled = compileGlb(codec);
            if (compiled == null) {
                callback({
                    ok: false,
                    message: 'The Blockbench glTF codec did not return binary content. Export the visual model manually as GLB to the configured path.'
                });
                return;
            }
            writeBinary(filePath, compiled);
            callback({ok: true});
        } catch (error) {
            callback({
                ok: false,
                message: `Could not export GLB automatically: ${error.message || error}. Export the visual model manually as GLB to the configured path.`
            });
        }
    }

    function gltfCodec() {
        if (typeof Codecs === 'undefined') return null;
        return Codecs.gltf || Codecs.glb || Codecs.GLTF || Object.values(Codecs).find(codec => {
            if (!codec) return false;
            const id = String(codec.id || '').toLowerCase();
            const extension = String(codec.extension || '').toLowerCase();
            const name = String(codec.name || '').toLowerCase();
            return id.includes('gltf') || extension === 'glb' || extension === 'gltf' || name.includes('gltf');
        });
    }

    function compileGlb(codec) {
        const optionSets = [
            {binary: true, embed_textures: true, include_animations: true},
            {format: 'glb', embed_textures: true, include_animations: true},
            {extension: 'glb', embed_textures: true, include_animations: true},
            {type: 'glb', embed_textures: true, include_animations: true},
            {}
        ];
        for (const options of optionSets) {
            try {
                const compiled = codec.compile(options);
                if (isBinaryContent(compiled)) return compiled;
                if (compiled && isBinaryContent(compiled.content)) return compiled.content;
                if (compiled && isBinaryContent(compiled.buffer)) return compiled.buffer;
            } catch (error) {
            }
        }
        return null;
    }

    function isBinaryContent(value) {
        return value instanceof ArrayBuffer
            || ArrayBuffer.isView(value)
            || (typeof Buffer !== 'undefined' && Buffer.isBuffer(value));
    }

    function writeBinary(filePath, content) {
        const fs = require('fs');
        if (typeof Buffer !== 'undefined' && Buffer.isBuffer(content)) {
            fs.writeFileSync(filePath, content);
        } else if (content instanceof ArrayBuffer) {
            fs.writeFileSync(filePath, Buffer.from(content));
        } else if (ArrayBuffer.isView(content)) {
            fs.writeFileSync(filePath, Buffer.from(content.buffer, content.byteOffset, content.byteLength));
        } else {
            throw new Error('compiled GLB content is not binary');
        }
    }

    function buildScene(visualPath) {
        const current = settings();
        const nodes = [];
        const groups = typeof Group !== 'undefined' ? Group.all : [];
        const cubes = typeof Cube !== 'undefined' ? Cube.all : [];

        groups.forEach(group => {
            if (group.export === false) return;
            nodes.push(nodeFromGroup(group, current.unit_scale));
        });
        cubes.forEach(cube => {
            if (cube.export === false) return;
            nodes.push(nodeFromCube(cube, current.unit_scale));
        });

        return {
            format: 'easycells3d.blockbench.scene',
            version: 1,
            model_name: Project && Project.name ? Project.name : 'Blockbench Scene',
            unit_scale: Number(current.unit_scale) || 16,
            visual: visualPath ? {
                model: relativeToProject(visualPath, current.project_root),
                kind: 'glb',
                match_by: ['easycells_id', 'blockbench_uuid', 'name']
            } : null,
            nodes
        };
    }

    function nodeBase(element) {
        ensureId(element);
        return {
            id: element.easycells_id,
            blockbench_uuid: element.uuid,
            name: element.name || element.uuid,
            parent: parentId(element),
            components: componentsOf(element)
        };
    }

    function nodeFromGroup(group, unitScale) {
        const node = nodeBase(group);
        node.kind = 'group';
        node.translation = vectorDiv(group.origin || [0, 0, 0], unitScale);
        node.rotation_euler_degrees = group.rotation || [0, 0, 0];
        node.scale = [1, 1, 1];
        return node;
    }

    function nodeFromCube(cube, unitScale) {
        const node = nodeBase(cube);
        node.kind = 'cube';
        node.translation = vectorDiv(cube.origin || cubeCenter(cube), unitScale);
        node.rotation_euler_degrees = cube.rotation || [0, 0, 0];
        node.scale = vectorDiv(cubeSize(cube), unitScale);
        node.cube = {
            from: vectorDiv(cube.from || [0, 0, 0], unitScale),
            to: vectorDiv(cube.to || [0, 0, 0], unitScale),
            origin: vectorDiv(cube.origin || cubeCenter(cube), unitScale)
        };
        return node;
    }

    function parentId(element) {
        const parent = element.parent;
        if (parent && parent !== 'root' && parent.uuid) {
            ensureId(parent);
            return parent.easycells_id;
        }
        return null;
    }

    function cubeCenter(cube) {
        return [
            ((cube.from || [0, 0, 0])[0] + (cube.to || [0, 0, 0])[0]) / 2,
            ((cube.from || [0, 0, 0])[1] + (cube.to || [0, 0, 0])[1]) / 2,
            ((cube.from || [0, 0, 0])[2] + (cube.to || [0, 0, 0])[2]) / 2
        ];
    }

    function cubeSize(cube) {
        return [
            Math.abs((cube.to || [0, 0, 0])[0] - (cube.from || [0, 0, 0])[0]),
            Math.abs((cube.to || [0, 0, 0])[1] - (cube.from || [0, 0, 0])[1]),
            Math.abs((cube.to || [0, 0, 0])[2] - (cube.from || [0, 0, 0])[2])
        ];
    }

    function defaultsForParameters(parameters) {
        const values = {};
        parameters.forEach(param => {
            values[param.name] = defaultFor(param);
        });
        return values;
    }

    function defaultsForFields(fields) {
        const values = {};
        Object.keys(fields).forEach(name => {
            values[name] = defaultFor(fields[name]);
        });
        return values;
    }

    function dialogField(metadata, value) {
        const type = fieldType(metadata);
        const field = {label: metadata.name || 'value', value: value === undefined ? defaultFor(metadata) : uiValue(value)};
        if (type === 'bool') field.type = 'checkbox';
        else if (type === 'int' || type === 'float') field.type = 'number';
        else field.type = 'text';
        return field;
    }

    function findParam(metadata, name) {
        return [].concat(metadata.required_args || [], metadata.optional_args || []).find(param => param.name === name) || {};
    }

    function defaultFor(metadata) {
        if (metadata && metadata.default !== undefined && metadata.default !== null) return metadata.default;
        const type = fieldType(metadata || {});
        if (type === 'bool') return false;
        if (type === 'int' || type === 'float') return 0;
        return '';
    }

    function fieldType(metadata) {
        return String((metadata && (metadata.ref || metadata.type)) || 'str');
    }

    function coerceValue(value, metadata) {
        const type = fieldType(metadata);
        if (type === 'bool') return !!value;
        if (type === 'int') return parseInt(value || 0, 10);
        if (type === 'float') return parseFloat(value || 0);
        return value;
    }

    function exportFieldValue(value, metadata) {
        const type = fieldType(metadata);
        if ((type === 'item' || type === 'component') && value) {
            return {$ref: String(value)};
        }
        return coerceValue(value, metadata);
    }

    function uiValue(value) {
        if (value && typeof value === 'object') return value.$id || value.$ref || '';
        return value;
    }

    function vectorDiv(vector, scalar) {
        scalar = Number(scalar) || 1;
        return [
            Number(vector[0] || 0) / scalar,
            Number(vector[1] || 0) / scalar,
            Number(vector[2] || 0) / scalar
        ];
    }

    function resolvePythonCommand(value, projectRoot) {
        if (!isDesktop()) return value || 'python';
        const fs = require('fs');
        const path = require('path');
        let raw = String(value || 'python').replace(/^"|"$/g, '');
        let candidate = path.isAbsolute(raw) ? raw : path.join(projectRoot, raw);
        if (fs.existsSync(candidate) && fs.statSync(candidate).isDirectory()) {
            const options = [
                path.join(candidate, 'Scripts', 'python.exe'),
                path.join(candidate, 'python.exe'),
                path.join(candidate, 'bin', 'python')
            ];
            for (const option of options) {
                if (fs.existsSync(option)) return option;
            }
        }
        if (fs.existsSync(candidate)) return candidate;
        return raw;
    }

    function resolveProjectPath(value, projectRoot) {
        if (!value) return '';
        if (!isDesktop()) return value;
        const path = require('path');
        const raw = String(value).replace(/^"|"$/g, '');
        return path.isAbsolute(raw) ? raw : path.join(projectRoot, raw);
    }

    function relativeToProject(filePath, projectRoot) {
        if (!isDesktop() || !filePath || !projectRoot) return filePath;
        const path = require('path');
        let relative = path.relative(projectRoot, filePath);
        if (!relative.startsWith('..') && !path.isAbsolute(relative)) {
            return relative.replace(/\\/g, '/');
        }
        return filePath.replace(/\\/g, '/');
    }

    function splitArgs(value) {
        if (!value) return [];
        return String(value).match(/(?:[^\s"]+|"[^"]*")+/g)?.map(arg => arg.replace(/^"|"$/g, '')) || [];
    }

    function isDesktop() {
        return typeof require === 'function' && Blockbench.platform !== 'web';
    }

    function createId() {
        if (typeof guid === 'function') return guid();
        if (typeof crypto !== 'undefined' && crypto.randomUUID) return crypto.randomUUID().replace(/-/g, '');
        return `ec3d_${Date.now().toString(36)}_${Math.random().toString(36).slice(2)}`;
    }

    function registerProperties() {
        easycellsIdCubeProperty = new Property(Cube, 'string', 'easycells_id', {default: ''});
        easycellsIdGroupProperty = new Property(Group, 'string', 'easycells_id', {default: ''});
        componentsCubeProperty = new Property(Cube, 'string', 'easycells_components', {default: '[]'});
        componentsGroupProperty = new Property(Group, 'string', 'easycells_components', {default: '[]'});
    }

    function makeAction(id, options) {
        const action = new Action(id, options);
        actions.push(action);
        return action;
    }

    Plugin.register(PLUGIN_ID, {
        title: 'EasyCells3D Components',
        author: 'PowerON0102',
        icon: 'extension',
        description: 'Configure EasyCells3D components on Blockbench cubes and groups.',
        version: '0.1.0',
        variant: 'both',
        onload() {
            registerProperties();
            makeAction('easycells3d_configure', {
                name: 'EasyCells3D Settings',
                icon: 'settings',
                click: configureDialog
            });
            makeAction('easycells3d_refresh_components', {
                name: 'Refresh EasyCells3D Components',
                icon: 'refresh',
                click: refreshComponents
            });
            makeAction('easycells3d_add_component', {
                name: 'Add EasyCells3D Component',
                icon: 'add',
                click: addComponentDialog
            });
            makeAction('easycells3d_edit_components', {
                name: 'Edit EasyCells3D Components',
                icon: 'edit',
                click: editComponentsDialog
            });
            makeAction('easycells3d_remove_component', {
                name: 'Remove EasyCells3D Component',
                icon: 'delete',
                click: removeComponentDialog
            });
            makeAction('easycells3d_export_scene', {
                name: 'Export EasyCells3D Scene',
                icon: 'archive',
                click: () => exportScene(false)
            });
            makeAction('easycells3d_export_run', {
                name: 'Export EasyCells3D Scene & Run',
                icon: 'play_arrow',
                click: () => exportScene(true)
            });

            MenuBar.addAction(actions[0], 'file');
            MenuBar.addAction(actions[1], 'file');
            MenuBar.addAction(actions[5], 'file.export');
            MenuBar.addAction(actions[6], 'file.export');
            MenuBar.addAction(actions[2], 'edit');
            MenuBar.addAction(actions[3], 'edit');
            MenuBar.addAction(actions[4], 'edit');
        },
        onunload() {
            actions.forEach(action => action.delete());
            actions = [];
            [easycellsIdCubeProperty, easycellsIdGroupProperty, componentsCubeProperty, componentsGroupProperty]
                .forEach(property => {
                    if (property && typeof property.delete === 'function') property.delete();
                });
        }
    });
})();
