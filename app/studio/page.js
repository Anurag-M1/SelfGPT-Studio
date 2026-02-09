'use client';

import { useState, useEffect, useCallback, useMemo, useRef } from 'react';
import dynamic from 'next/dynamic';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardDescription, CardHeader, CardTitle } from '@/components/ui/card';
import { Input } from '@/components/ui/input';
import { Badge } from '@/components/ui/badge';
import { ScrollArea } from '@/components/ui/scroll-area';
import { Separator } from '@/components/ui/separator';
import { Tabs, TabsContent, TabsList, TabsTrigger } from '@/components/ui/tabs';
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from '@/components/ui/select';
import {
  ResizablePanelGroup,
  ResizablePanel,
  ResizableHandle,
} from '@/components/ui/resizable';
import {
  Code2, Eye, Send, Plus, LogOut, Trash2,
  Loader2, Sparkles, Globe, ArrowLeft,
  User, Mail, Lock, Wand2, ExternalLink, Copy, Check,
  FolderOpen, Pencil, MessageSquare, Zap, LayoutDashboard,
  RefreshCw, Save, Share2, Terminal, Search, FilePlus2, Settings, X,
  Bot, Package, GitCompare, Upload, FileArchive, Github, Instagram
} from 'lucide-react';

// Dynamic import for Monaco Editor
const MonacoEditor = dynamic(() => import('@monaco-editor/react'), {
  ssr: false,
  loading: () => (
    <div className="flex items-center justify-center h-full bg-[#1e1e1e]">
      <Loader2 className="w-8 h-8 animate-spin text-violet-500" />
    </div>
  )
});

// --- API Helper ---
const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL || '';
const api = {
  async request(path, options = {}) {
    const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
    const headers = {
      'Content-Type': 'application/json',
      ...(token ? { Authorization: `Bearer ${token}` } : {}),
      ...options.headers,
    };

    const response = await fetch(`${API_BASE}/api/${path}`, { ...options, headers });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Request failed');
    return data;
  },
  register: (d) => api.request('auth/register', { method: 'POST', body: JSON.stringify(d) }),
  login: (d) => api.request('auth/login', { method: 'POST', body: JSON.stringify(d) }),
  getMe: () => api.request('auth/me'),
  createProject: (d) => api.request('projects', { method: 'POST', body: JSON.stringify(d) }),
  listProjects: () => api.request('projects'),
  getProject: (id) => api.request(`projects/${id}`),
  updateProject: (id, d) => api.request(`projects/${id}`, { method: 'PUT', body: JSON.stringify(d) }),
  deleteProject: (id) => api.request(`projects/${id}`, { method: 'DELETE' }),
  generate: (d) => api.request('generate', { method: 'POST', body: JSON.stringify(d) }),
  llmOptions: () => api.request('llm/options'),
  llmModels: (provider) => api.request(`llm/models?provider=${encodeURIComponent(provider)}`),
  templates: () => api.request('templates'),
  runProfiles: () => api.request('run/profiles'),
  agents: () => api.request('agents'),
  listSnapshots: (projectId) => api.request(`snapshots?projectId=${encodeURIComponent(projectId)}`),
  createSnapshot: (d) => api.request('snapshots', { method: 'POST', body: JSON.stringify(d) }),
  restoreSnapshot: (snapshotId) => api.request(`snapshots/${snapshotId}/restore`, { method: 'POST' }),
  getSnapshot: (snapshotId) => api.request(`snapshots/${snapshotId}`),
  exportProject: async (projectId) => {
    const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const response = await fetch(`${API_BASE}/api/projects/${projectId}/export`, { headers });
    if (!response.ok) throw new Error('Export failed');
    return response.blob();
  },
  importProject: async (projectId, file) => {
    const token = typeof window !== 'undefined' ? localStorage.getItem('token') : null;
    const headers = token ? { Authorization: `Bearer ${token}` } : {};
    const formData = new FormData();
    formData.append('file', file);
    const response = await fetch(`${API_BASE}/api/projects/${projectId}/import`, {
      method: 'POST',
      headers,
      body: formData,
    });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Import failed');
    return data;
  },
};

const LogoMark = ({ className = '' }) => (
  <span className={`font-bold tracking-tight text-white ${className}`}>SG</span>
);

const STACK_OPTIONS = [
  { id: 'web', label: 'Static Web' },
  { id: 'react', label: 'React' },
  { id: 'nextjs', label: 'Next.js' },
  { id: 'node', label: 'Node' },
  { id: 'mern', label: 'MERN' },
  { id: 'python', label: 'Python' },
  { id: 'fastapi', label: 'FastAPI' },
  { id: 'flask', label: 'Flask' },
  { id: 'django', label: 'Django' },
];

// --- Auth View ---
function AuthView({ onAuth }) {
  const [isLogin, setIsLogin] = useState(true);
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [name, setName] = useState('');
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');

  const handleSubmit = async (e) => {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      const data = isLogin
        ? await api.login({ email, password })
        : await api.register({ email, password, name });
      localStorage.setItem('token', data.token);
      onAuth(data.user);
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex items-center justify-center relative overflow-hidden">
      {/* Animated background */}
      <div className="absolute inset-0 bg-gradient-to-br from-slate-950 via-violet-950/50 to-slate-950">
        <div className="absolute top-1/4 left-1/4 w-96 h-96 bg-violet-600/20 rounded-full blur-3xl animate-pulse" />
        <div className="absolute bottom-1/4 right-1/4 w-96 h-96 bg-indigo-600/20 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '1s' }} />
        <div className="absolute top-1/2 left-1/2 w-64 h-64 bg-purple-600/10 rounded-full blur-3xl animate-pulse" style={{ animationDelay: '2s' }} />
      </div>

      <div className="relative z-10 w-full max-w-md px-4">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center gap-3 mb-4">
            <div className="w-12 h-12 rounded-xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center shadow-lg shadow-violet-500/25">
              <LogoMark className="text-lg" />
            </div>
            <h1 className="text-3xl font-bold bg-gradient-to-r from-violet-400 to-indigo-400 bg-clip-text text-transparent">
              SelfGPT Studio
            </h1>
          </div>
          <p className="text-slate-400 text-sm">Build, run, and deploy apps with SelfGPT</p>
        </div>

        <Card className="border-slate-800/50 bg-slate-900/80 backdrop-blur-xl shadow-2xl shadow-violet-500/5">
          <CardHeader className="space-y-1 pb-4">
            <CardTitle className="text-xl text-center text-slate-100">
              {isLogin ? 'Welcome back' : 'Create account'}
            </CardTitle>
            <CardDescription className="text-center text-slate-400">
              {isLogin ? 'Sign in to your account' : 'Get started for free'}
            </CardDescription>
          </CardHeader>
          <CardContent>
            <form onSubmit={handleSubmit} className="space-y-4">
              {!isLogin && (
                <div className="relative">
                  <User className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                  <Input
                    placeholder="Full Name"
                    value={name}
                    onChange={(e) => setName(e.target.value)}
                    className="pl-10 bg-slate-800/50 border-slate-700 focus:border-violet-500 text-slate-100 placeholder:text-slate-500"
                    required={!isLogin}
                  />
                </div>
              )}
              <div className="relative">
                <Mail className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <Input
                  type="email"
                  placeholder="Email"
                  value={email}
                  onChange={(e) => setEmail(e.target.value)}
                  className="pl-10 bg-slate-800/50 border-slate-700 focus:border-violet-500 text-slate-100 placeholder:text-slate-500"
                  required
                />
              </div>
              <div className="relative">
                <Lock className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
                <Input
                  type="password"
                  placeholder="Password"
                  value={password}
                  onChange={(e) => setPassword(e.target.value)}
                  className="pl-10 bg-slate-800/50 border-slate-700 focus:border-violet-500 text-slate-100 placeholder:text-slate-500"
                  required
                  minLength={6}
                />
              </div>

              {error && (
                <div className="text-red-400 text-sm bg-red-500/10 border border-red-500/20 rounded-lg px-3 py-2">
                  {error}
                </div>
              )}

              <Button
                type="submit"
                disabled={loading}
                className="w-full bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white font-medium shadow-lg shadow-violet-500/25 transition-all duration-200"
              >
                {loading ? (
                  <Loader2 className="w-4 h-4 animate-spin mr-2" />
                ) : isLogin ? (
                  'Sign In'
                ) : (
                  'Create Account'
                )}
              </Button>
            </form>

            <div className="mt-6 text-center">
              <button
                onClick={() => { setIsLogin(!isLogin); setError(''); }}
                className="text-sm text-slate-400 hover:text-violet-400 transition-colors"
              >
                {isLogin ? "Don't have an account? Sign up" : 'Already have an account? Sign in'}
              </button>
            </div>
          </CardContent>
        </Card>
      </div>
    </div>
  );
}

// --- Dashboard View ---
function DashboardView({ user, onLogout, onOpenProject, onCreateProject }) {
  const [projects, setProjects] = useState([]);
  const [loading, setLoading] = useState(true);
  const [creating, setCreating] = useState(false);
  const [newProjectName, setNewProjectName] = useState('');
  const [showCreate, setShowCreate] = useState(false);
  const [templates, setTemplates] = useState([]);
  const [selectedTemplate, setSelectedTemplate] = useState('blank');
  const [templatesLoading, setTemplatesLoading] = useState(false);

  useEffect(() => {
    loadProjects();
  }, []);

  useEffect(() => {
    if (!showCreate) return;
    setTemplatesLoading(true);
    api.templates()
      .then((data) => {
        const list = data.templates || [];
        setTemplates(list);
        if (list.length > 0 && !list.find(t => t.id === selectedTemplate)) {
          setSelectedTemplate(list[0].id);
        }
      })
      .catch(() => {
        setTemplates([]);
      })
      .finally(() => setTemplatesLoading(false));
  }, [showCreate]);

  const loadProjects = async () => {
    try {
      const data = await api.listProjects();
      setProjects(data.projects || []);
    } catch (err) {
      console.error('Failed to load projects:', err);
    } finally {
      setLoading(false);
    }
  };

  const handleCreate = async () => {
    if (!newProjectName.trim()) return;
    setCreating(true);
    try {
      const template = templates.find(t => t.id === selectedTemplate);
      const runtime = template?.runtime || 'web';
      const stackHint = template?.id?.includes('next') ? 'nextjs'
        : template?.id?.includes('react') ? 'react'
        : template?.id?.includes('mern') ? 'mern'
        : template?.id?.includes('fastapi') ? 'fastapi'
        : template?.id?.includes('flask') ? 'flask'
        : template?.id?.includes('django') ? 'django'
        : runtime === 'python' ? 'python'
        : runtime === 'node' ? 'node'
        : 'web';
      const data = await api.createProject({
        name: newProjectName.trim(),
        templateId: selectedTemplate,
        settings: { stack: stackHint },
      });
      setShowCreate(false);
      setNewProjectName('');
      onOpenProject(data.project);
    } catch (err) {
      console.error('Failed to create project:', err);
    } finally {
      setCreating(false);
    }
  };

  const handleDelete = async (projectId, e) => {
    e.stopPropagation();
    if (!confirm('Delete this project?')) return;
    try {
      await api.deleteProject(projectId);
      setProjects(prev => prev.filter(p => p.id !== projectId));
    } catch (err) {
      console.error('Failed to delete project:', err);
    }
  };

  return (
    <div className="min-h-screen bg-gradient-to-b from-slate-950 via-slate-900 to-slate-950 flex flex-col">
      {/* Navbar */}
      <nav className="border-b border-slate-800/50 bg-slate-950/80 backdrop-blur-xl sticky top-0 z-50">
        <div className="max-w-7xl mx-auto px-6 h-16 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-9 h-9 rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center">
              <LogoMark className="text-sm" />
            </div>
            <span className="text-lg font-bold bg-gradient-to-r from-violet-400 to-indigo-400 bg-clip-text text-transparent">
              SelfGPT Studio
            </span>
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2 text-sm text-slate-400">
              <div className="w-8 h-8 rounded-full bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center text-white text-xs font-bold">
                {user?.name?.charAt(0).toUpperCase()}
              </div>
              <span className="hidden sm:inline">{user?.name}</span>
            </div>
            <Button
              variant="ghost"
              size="sm"
              onClick={onLogout}
              className="text-slate-400 hover:text-red-400 hover:bg-red-500/10"
            >
              <LogOut className="w-4 h-4" />
            </Button>
          </div>
        </div>
      </nav>

      {/* Content */}
      <div className="max-w-7xl mx-auto px-6 py-12 flex-1 flex flex-col">
        {/* Header */}
        <div className="flex items-center justify-between mb-10">
          <div>
            <h2 className="text-3xl font-bold text-slate-100 mb-2">Your Projects</h2>
            <p className="text-slate-400">Build, edit, and run projects with SelfGPT</p>
          </div>
          <Button
            onClick={() => setShowCreate(true)}
            className="bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white shadow-lg shadow-violet-500/25"
          >
            <Plus className="w-4 h-4 mr-2" />
            New Project
          </Button>
        </div>

        {/* Create Project Modal */}
        {showCreate && (
          <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/60 backdrop-blur-sm">
            <Card className="w-full max-w-md border-slate-800/50 bg-slate-900/95 shadow-2xl animate-fade-in">
              <CardHeader>
                <CardTitle className="text-slate-100">Create New Project</CardTitle>
                <CardDescription className="text-slate-400">Choose a template and name your project</CardDescription>
              </CardHeader>
              <CardContent className="space-y-4">
                <div>
                  <p className="text-xs text-slate-500 mb-2">Template</p>
                  <div className="grid grid-cols-1 sm:grid-cols-2 gap-2">
                    {templatesLoading ? (
                      <div className="text-xs text-slate-500">Loading templates...</div>
                    ) : (
                      (templates.length ? templates : [{ id: 'blank', name: 'Blank', description: 'Minimal starter', runtime: 'web' }]).map((tpl) => (
                        <button
                          key={tpl.id}
                          onClick={() => setSelectedTemplate(tpl.id)}
                          className={`text-left p-2 rounded border text-xs transition-all ${
                            selectedTemplate === tpl.id
                              ? 'border-violet-500/60 bg-violet-500/10 text-slate-100'
                              : 'border-slate-800/60 bg-slate-900/60 text-slate-400 hover:border-slate-700'
                          }`}
                        >
                          <div className="font-medium">{tpl.name}</div>
                          <div className="text-[10px] text-slate-500">{tpl.description}</div>
                        </button>
                      ))
                    )}
                  </div>
                </div>
                <Input
                  placeholder="Project name"
                  value={newProjectName}
                  onChange={(e) => setNewProjectName(e.target.value)}
                  onKeyDown={(e) => e.key === 'Enter' && handleCreate()}
                  className="bg-slate-800/50 border-slate-700 focus:border-violet-500 text-slate-100 placeholder:text-slate-500"
                  autoFocus
                />
                <div className="flex gap-3 justify-end">
                  <Button
                    variant="ghost"
                    onClick={() => { setShowCreate(false); setNewProjectName(''); }}
                    className="text-slate-400"
                  >
                    Cancel
                  </Button>
                  <Button
                    onClick={handleCreate}
                    disabled={!newProjectName.trim() || creating}
                    className="bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white"
                  >
                    {creating ? <Loader2 className="w-4 h-4 animate-spin mr-2" /> : <Sparkles className="w-4 h-4 mr-2" />}
                    Create
                  </Button>
                </div>
              </CardContent>
            </Card>
          </div>
        )}

        {/* Projects Grid */}
        {loading ? (
          <div className="flex items-center justify-center py-20">
            <Loader2 className="w-8 h-8 animate-spin text-violet-500" />
          </div>
        ) : projects.length === 0 ? (
          <div className="text-center py-20">
            <div className="w-20 h-20 rounded-2xl bg-slate-800/50 flex items-center justify-center mx-auto mb-6">
              <FolderOpen className="w-10 h-10 text-slate-600" />
            </div>
            <h3 className="text-xl font-semibold text-slate-300 mb-2">No projects yet</h3>
            <p className="text-slate-500 mb-6">Create your first project and start building with AI</p>
            <Button
              onClick={() => setShowCreate(true)}
              className="bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white"
            >
              <Plus className="w-4 h-4 mr-2" />
              Create Your First Project
            </Button>
          </div>
        ) : (
          <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-6">
            {projects.map((project) => (
              <Card
                key={project.id}
                onClick={() => onOpenProject(project)}
                className="border-slate-800/50 bg-slate-900/50 hover:bg-slate-800/50 hover:border-violet-500/30 cursor-pointer transition-all duration-200 group"
              >
                <CardHeader className="pb-3">
                  <div className="flex items-start justify-between">
                    <div className="w-10 h-10 rounded-lg bg-gradient-to-br from-violet-500/20 to-indigo-500/20 flex items-center justify-center group-hover:from-violet-500/30 group-hover:to-indigo-500/30 transition-colors">
                      <Globe className="w-5 h-5 text-violet-400" />
                    </div>
                    <Button
                      variant="ghost"
                      size="icon"
                      onClick={(e) => handleDelete(project.id, e)}
                      className="opacity-0 group-hover:opacity-100 text-slate-500 hover:text-red-400 hover:bg-red-500/10 transition-all h-8 w-8"
                    >
                      <Trash2 className="w-4 h-4" />
                    </Button>
                  </div>
                  <CardTitle className="text-slate-100 text-lg mt-3">{project.name}</CardTitle>
                </CardHeader>
                <CardContent>
                  <div className="flex items-center gap-4 text-xs text-slate-500">
                    <span className="flex items-center gap-1">
                      <MessageSquare className="w-3 h-3" />
                      {project.messageCount || 0} prompts
                    </span>
                    <span>
                      {new Date(project.updatedAt).toLocaleDateString()}
                    </span>
                  </div>
                </CardContent>
              </Card>
            ))}
          </div>
        )}

        <footer className="mt-auto border-t border-slate-800/60 pt-6 flex flex-col md:flex-row items-center md:items-center md:justify-between gap-4 text-sm text-slate-400 text-center md:text-left">
          <div className="w-full md:w-auto">
            <p className="text-slate-300 font-medium">Designed & developed by Anurag</p>
          </div>
          <div className="flex flex-col sm:flex-row items-center sm:items-center gap-3 w-full md:w-auto md:justify-end">
            <a
              href="https://github.com/anurag-m1"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 text-slate-300 hover:text-white transition"
            >
              <Github className="w-4 h-4" />
              <span className="text-xs">github.com/anurag-m1</span>
            </a>
            <a
              href="https://instagram.com/ca_anuragsingh"
              target="_blank"
              rel="noreferrer"
              className="inline-flex items-center gap-2 text-slate-300 hover:text-white transition"
            >
              <Instagram className="w-4 h-4" />
              <span className="text-xs">instagram.com/ca_anuragsingh</span>
            </a>
          </div>
        </footer>
      </div>
    </div>
  );
}

// --- Editor View ---
function EditorView({ project: initialProject, onBack }) {
  const [code, setCode] = useState(initialProject?.code || '');
  const [messages, setMessages] = useState(initialProject?.messages || []);
  const [prompt, setPrompt] = useState('');
  const [generating, setGenerating] = useState(false);
  const [saving, setSaving] = useState(false);
  const [saved, setSaved] = useState(false);
  const [projectName, setProjectName] = useState(initialProject?.name || 'Untitled');
  const [previewKey, setPreviewKey] = useState(0);
  const [copied, setCopied] = useState(false);
  const [showPreview, setShowPreview] = useState(true);
  const [llmOptions, setLlmOptions] = useState(null);
  const [llmProvider, setLlmProvider] = useState('');
  const [llmModel, setLlmModel] = useState(process.env.NEXT_PUBLIC_LLM_MODEL || '');
  const [llmLoading, setLlmLoading] = useState(false);
  const [llmModels, setLlmModels] = useState([]);
  const [llmModelsLoading, setLlmModelsLoading] = useState(false);
  const [files, setFiles] = useState([]);
  const [openFiles, setOpenFiles] = useState([]);
  const [activeFile, setActiveFile] = useState('');
  const [fileSearch, setFileSearch] = useState('');
  const [envVars, setEnvVars] = useState([]);
  const [stack, setStack] = useState(initialProject?.settings?.stack || 'web');
  const [runProfiles, setRunProfiles] = useState([]);
  const [runProfileId, setRunProfileId] = useState(initialProject?.settings?.runProfileId || '');
  const [agents, setAgents] = useState([]);
  const [activeAgent, setActiveAgent] = useState('builder');
  const [terminalCommand, setTerminalCommand] = useState('');
  const [terminalLogs, setTerminalLogs] = useState('');
  const [terminalStatus, setTerminalStatus] = useState('disconnected');
  const [terminalStatusDetail, setTerminalStatusDetail] = useState('');
  const [bottomTab, setBottomTab] = useState('terminal');
  const [snapshots, setSnapshots] = useState([]);
  const [snapshotsLoading, setSnapshotsLoading] = useState(false);
  const [diffSnapshotId, setDiffSnapshotId] = useState('');
  const [diffSnapshot, setDiffSnapshot] = useState(null);
  const [diffFilePath, setDiffFilePath] = useState('');
  const [insights, setInsights] = useState([]);
  const [npmPackageInput, setNpmPackageInput] = useState('');
  const [pipPackageInput, setPipPackageInput] = useState('');
  const [importing, setImporting] = useState(false);
  const [exporting, setExporting] = useState(false);
  const chatEndRef = useRef(null);
  const iframeRef = useRef(null);
  const wsRef = useRef(null);
  const clientIdRef = useRef((typeof crypto !== 'undefined' && crypto.randomUUID) ? crypto.randomUUID() : `${Date.now()}`);
  const broadcastTimerRef = useRef(null);
  const activeFileRef = useRef('');
  const terminalSocketRef = useRef(null);
  const terminalReconnectRef = useRef(null);
  const importInputRef = useRef(null);

  useEffect(() => {
    if (chatEndRef.current) {
      chatEndRef.current.scrollIntoView({ behavior: 'smooth' });
    }
  }, [messages]);

  useEffect(() => {
    activeFileRef.current = activeFile;
  }, [activeFile]);

  const getLanguageFromPath = (path) => {
    if (!path) return 'plaintext';
    const ext = path.split('.').pop().toLowerCase();
    if (ext === 'js' || ext === 'jsx') return 'javascript';
    if (ext === 'ts' || ext === 'tsx') return 'typescript';
    if (ext === 'css') return 'css';
    if (ext === 'json') return 'json';
    if (ext === 'py') return 'python';
    if (ext === 'html' || ext === 'htm') return 'html';
    if (ext === 'md') return 'markdown';
    return 'plaintext';
  };

  const normalizeFiles = (project) => {
    if (project?.files && project.files.length > 0) return project.files;
    const fallback = project?.code || '';
    return [{
      path: 'index.html',
      content: fallback || '<!DOCTYPE html>\\n<html><head></head><body></body></html>',
      language: 'html',
    }];
  };

  const getFileByPath = (path) => files.find(f => f.path === path);

  const indexHtmlContent = (() => {
    const file = files.find(f => f.path === 'index.html');
    return file?.content || '';
  })();

  const previewHtml = useMemo(() => {
    const baseHtml = indexHtmlContent || '';
    if (!baseHtml) {
      return (
        '<!DOCTYPE html><html><head><meta charset="UTF-8" />' +
        '<meta name="viewport" content="width=device-width, initial-scale=1.0" />' +
        '<title>Preview</title></head><body>' +
        '<h2 style="font-family:system-ui;">No preview available</h2>' +
        '<p style="font-family:system-ui;">This project does not include an index.html file.</p>' +
        '</body></html>'
      );
    }

    const fileMap = new Map(files.map(f => [f.path, f.content || '']));
    const normalize = (raw) => {
      if (!raw) return '';
      return raw.split('?')[0].split('#')[0].replace(/^\.\//, '').replace(/^\//, '');
    };

    let html = baseHtml;
    let cssInlined = false;
    let jsInlined = false;

    html = html.replace(/<link[^>]+href=["']([^"']+)["'][^>]*>/gi, (match, href) => {
      const path = normalize(href);
      if (fileMap.has(path)) {
        cssInlined = true;
        return `<style>${fileMap.get(path)}</style>`;
      }
      return match;
    });

    html = html.replace(/<script[^>]+src=["']([^"']+)["'][^>]*>\s*<\/script>/gi, (match, src) => {
      const path = normalize(src);
      if (fileMap.has(path)) {
        jsInlined = true;
        return `<script>${fileMap.get(path)}</script>`;
      }
      return match;
    });

    if (!cssInlined && fileMap.has('styles.css')) {
      const styleTag = `<style>${fileMap.get('styles.css')}</style>`;
      html = html.includes('</head>')
        ? html.replace('</head>', `${styleTag}\n</head>`)
        : `${styleTag}\n${html}`;
    }

    if (!jsInlined && fileMap.has('script.js')) {
      const scriptTag = `<script>${fileMap.get('script.js')}</script>`;
      html = html.includes('</body>')
        ? html.replace('</body>', `${scriptTag}\n</body>`)
        : `${html}\n${scriptTag}`;
    }

    return html;
  }, [files, indexHtmlContent]);

  const packageJsonFile = useMemo(() => files.find(f => f.path === 'package.json'), [files]);
  const requirementsFile = useMemo(() => files.find(f => f.path === 'requirements.txt'), [files]);

  const packageDeps = useMemo(() => {
    if (!packageJsonFile?.content) return {};
    try {
      const json = JSON.parse(packageJsonFile.content);
      return json.dependencies || {};
    } catch (err) {
      return {};
    }
  }, [packageJsonFile]);

  const requirementsList = useMemo(() => {
    if (!requirementsFile?.content) return [];
    return requirementsFile.content
      .split('\n')
      .map(line => line.trim())
      .filter(line => line && !line.startsWith('#'));
  }, [requirementsFile]);

  // Load full project data
  useEffect(() => {
    if (initialProject?.id) {
      api.getProject(initialProject.id).then(data => {
        if (data.project) {
          const projectFiles = normalizeFiles(data.project);
          const settings = data.project.settings || {};
          setFiles(projectFiles);
          setOpenFiles(projectFiles.length ? [projectFiles[0].path] : []);
          setActiveFile(projectFiles.length ? projectFiles[0].path : '');
          setCode(projectFiles.length ? (projectFiles[0].content || '') : (data.project.code || ''));
          setMessages(data.project.messages || []);
          setProjectName(data.project.name || 'Untitled');
          setEnvVars(data.project.envVars || []);
          setStack(settings.stack || 'web');
          setRunProfileId(settings.runProfileId || '');
          setActiveAgent(settings.agent || 'builder');
        }
      }).catch(console.error);
    }
  }, [initialProject?.id]);

  useEffect(() => {
    if (!initialProject?.id) return;
    setSnapshotsLoading(true);
    api.listSnapshots(initialProject.id)
      .then((data) => setSnapshots(data.snapshots || []))
      .catch(() => setSnapshots([]))
      .finally(() => setSnapshotsLoading(false));
  }, [initialProject?.id]);

  useEffect(() => {
    if (!diffSnapshotId) {
      setDiffSnapshot(null);
      setDiffFilePath('');
      return;
    }
    api.getSnapshot(diffSnapshotId)
      .then((data) => {
        const snap = data.snapshot;
        setDiffSnapshot(snap);
        if (snap?.files?.length) {
          setDiffFilePath(prev => prev || snap.files[0].path);
        }
      })
      .catch(() => {
        setDiffSnapshot(null);
      });
  }, [diffSnapshotId]);

  useEffect(() => {
    if (!files.length && initialProject) {
      const projectFiles = normalizeFiles(initialProject);
      const settings = initialProject?.settings || {};
      setFiles(projectFiles);
      setOpenFiles(projectFiles.length ? [projectFiles[0].path] : []);
      setActiveFile(projectFiles.length ? projectFiles[0].path : '');
      setCode(projectFiles.length ? (projectFiles[0].content || '') : (initialProject?.code || ''));
      setEnvVars(initialProject?.envVars || []);
      setStack(settings.stack || 'web');
      setRunProfileId(settings.runProfileId || '');
      setActiveAgent(settings.agent || 'builder');
    }
  }, [initialProject]);

  // Load LLM options
  useEffect(() => {
    let mounted = true;
    setLlmLoading(true);
    api.llmOptions()
      .then((data) => {
        if (!mounted) return;
        setLlmOptions(data);
        const enabled = (data?.providers || []).filter(p => p.enabled).map(p => p.id);
        if (enabled.length > 0) {
          const preferred = data?.defaultProvider && enabled.includes(data.defaultProvider)
            ? data.defaultProvider
            : enabled[0];
          setLlmProvider((prev) => prev || preferred);
        } else if (data?.defaultProvider) {
          setLlmProvider((prev) => prev || data.defaultProvider);
        }
        if (data?.defaultModel) {
          setLlmModel((prev) => prev || data.defaultModel);
        }
      })
      .catch(() => {
        if (!mounted) return;
        setLlmOptions(null);
      })
      .finally(() => {
        if (!mounted) return;
        setLlmLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, []);

  useEffect(() => {
    let mounted = true;
    api.runProfiles()
      .then((data) => {
        if (!mounted) return;
        const profiles = data.profiles || [];
        setRunProfiles(profiles);
        if (profiles.length) {
          const exists = profiles.some(p => p.id === runProfileId);
          if (!runProfileId || !exists) {
            const match = profiles.find(p => p.stack === stack) || profiles[0];
            if (match) setRunProfileId(match.id);
          }
        }
      })
      .catch(() => {
        if (!mounted) return;
        setRunProfiles([]);
      });
    return () => {
      mounted = false;
    };
  }, [stack, runProfileId]);

  useEffect(() => {
    let mounted = true;
    api.agents()
      .then((data) => {
        if (!mounted) return;
        setAgents(data.agents || []);
        if (!activeAgent && data.agents?.length) {
          setActiveAgent(data.agents[0].id);
        }
      })
      .catch(() => {
        if (!mounted) return;
        setAgents([]);
      });
    return () => {
      mounted = false;
    };
  }, []);

  // Collaboration WebSocket
  useEffect(() => {
    if (!initialProject?.id) return;
    const baseUrl = typeof window !== 'undefined' ? window.location.origin : '';
    const apiBase = API_BASE || baseUrl;
    const wsUrl = apiBase.replace(/^http/, 'ws') + `/ws/projects/${initialProject.id}`;
    const ws = new WebSocket(wsUrl);
    wsRef.current = ws;

    ws.onmessage = (event) => {
      try {
        const msg = JSON.parse(event.data);
        if (msg.clientId && msg.clientId === clientIdRef.current) return;
        if (msg.type === 'file_update') {
          setFiles(prev => prev.map(f => f.path === msg.path ? { ...f, content: msg.content } : f));
          if (msg.path === activeFileRef.current) {
            setCode(msg.content || '');
          }
        }
        if (msg.type === 'file_create') {
          setFiles(prev => prev.some(f => f.path === msg.path) ? prev : [...prev, msg.file]);
        }
        if (msg.type === 'file_delete') {
          setFiles(prev => prev.filter(f => f.path !== msg.path));
          setOpenFiles(prev => prev.filter(p => p !== msg.path));
          setActiveFile(prev => (prev === msg.path ? '' : prev));
        }
        if (msg.type === 'file_rename') {
          setFiles(prev => prev.map(f => f.path === msg.oldPath ? { ...f, path: msg.newPath } : f));
          setOpenFiles(prev => prev.map(p => p === msg.oldPath ? msg.newPath : p));
          setActiveFile(prev => (prev === msg.oldPath ? msg.newPath : prev));
        }
      } catch (e) {
        // ignore malformed messages
      }
    };

    return () => {
      ws.close();
    };
  }, [initialProject?.id]);

  useEffect(() => {
    if (!initialProject?.id) return;
    let closed = false;
    let attemptedWithoutToken = false;
    const baseUrl = typeof window !== 'undefined' ? window.location.origin : '';
    const apiBase = API_BASE || baseUrl;
    const token = typeof window !== 'undefined' ? localStorage.getItem('token') : '';

    const connect = (withToken) => {
      if (closed) return;
      setTerminalStatus('connecting');
      setTerminalStatusDetail('');
      const wsUrl = apiBase.replace(/^http/, 'ws') + `/ws/terminal/${initialProject.id}${withToken && token ? `?token=${encodeURIComponent(token)}` : ''}`;
      const ws = new WebSocket(wsUrl);
      terminalSocketRef.current = ws;

      ws.onopen = () => {
        if (closed) return;
        setTerminalStatus('connected');
        setTerminalStatusDetail(withToken && token ? 'authenticated' : 'local');
        setTerminalLogs(prev => prev + '\n[terminal connected]\n');
        try {
          ws.send('\n');
        } catch (err) {
          // ignore send errors
        }
      };

      ws.onmessage = (event) => {
        setTerminalLogs(prev => prev + event.data);
      };

      ws.onerror = () => {
        if (closed) return;
        setTerminalStatus('error');
      };

      ws.onclose = (event) => {
        if (closed) return;
        if (withToken && token && !attemptedWithoutToken) {
          attemptedWithoutToken = true;
          connect(false);
          return;
        }
        setTerminalStatus('disconnected');
        setTerminalStatusDetail(event?.reason || `code ${event?.code || 'unknown'}`);
      };
    };

    terminalReconnectRef.current = () => {
      if (terminalSocketRef.current) {
        try {
          terminalSocketRef.current.close();
        } catch (err) {
          // ignore
        }
      }
      attemptedWithoutToken = false;
      connect(!!token);
    };

    connect(!!token);

    return () => {
      closed = true;
      if (terminalSocketRef.current) {
        terminalSocketRef.current.close();
      }
    };
  }, [initialProject?.id]);

  useEffect(() => {
    const logs = terminalLogs.slice(-6000);
    const parsed = [];
    const moduleNotFound = logs.match(/ModuleNotFoundError: No module named ['\"]([^'\"]+)['\"]/);
    if (moduleNotFound) {
      parsed.push({
        title: `Python module missing: ${moduleNotFound[1]}`,
        detail: 'Python cannot find the module during execution.',
        suggestion: `pip install ${moduleNotFound[1]}`,
      });
    }
    const nodeMissing = logs.match(/Cannot find module ['\"]([^'\"]+)['\"]/);
    if (nodeMissing) {
      parsed.push({
        title: `Node module missing: ${nodeMissing[1]}`,
        detail: 'Node cannot resolve this package.',
        suggestion: `npm install ${nodeMissing[1]}`,
      });
    }
    if (/EADDRINUSE/.test(logs)) {
      parsed.push({
        title: 'Port already in use',
        detail: 'Another process is already bound to the port.',
        suggestion: 'Stop the other process or change the port.',
      });
    }
    if (/Traceback \\(most recent call last\\):/.test(logs)) {
      parsed.push({
        title: 'Python traceback detected',
        detail: 'A runtime error occurred. Check the last lines for the root cause.',
      });
    }
    if (/SyntaxError/.test(logs)) {
      parsed.push({
        title: 'Syntax error',
        detail: 'There is a syntax error in your code.',
        suggestion: 'Open the file and fix the reported line.',
      });
    }
    setInsights(parsed);
  }, [terminalLogs]);

  // Load models for selected provider
  useEffect(() => {
    if (!llmProvider) return;
    let mounted = true;
    setLlmModelsLoading(true);
    api.llmModels(llmProvider)
      .then((data) => {
        if (!mounted) return;
        const models = data?.models || [];
        setLlmModels(models);
        if (models.length > 0) {
          const modelIds = models.map(m => m.id);
          if (!llmModel || !modelIds.includes(llmModel)) {
            setLlmModel(modelIds[0]);
          }
        }
      })
      .catch(() => {
        if (!mounted) return;
        setLlmModels([]);
      })
      .finally(() => {
        if (!mounted) return;
        setLlmModelsLoading(false);
      });
    return () => {
      mounted = false;
    };
  }, [llmProvider]);

  const providerLabel = (providerId) => {
    switch ((providerId || '').toLowerCase()) {
      case 'groq':
        return 'Groq';
      case 'openai':
        return 'OpenAI';
      case 'mistral':
        return 'Mistral';
      case 'deepseek':
        return 'DeepSeek';
      case 'gemini':
        return 'Gemini';
      default:
        return providerId || 'LLM';
    }
  };

  const broadcast = (payload) => {
    if (wsRef.current && wsRef.current.readyState === 1) {
      wsRef.current.send(JSON.stringify({ ...payload, clientId: clientIdRef.current }));
    }
  };

  const applyFileUpdate = (path, content) => {
    if (!path) return;
    const exists = files.find(f => f.path === path);
    const language = getLanguageFromPath(path);
    if (exists) {
      setFiles(prev => prev.map(f => f.path === path ? { ...f, content, language: f.language || language } : f));
      broadcast({ type: 'file_update', path, content });
    } else {
      const newFile = { path, content, language };
      setFiles(prev => [...prev, newFile]);
      broadcast({ type: 'file_create', path, file: newFile });
    }
    if (activeFileRef.current === path) {
      setCode(content);
    }
  };

  const openFile = (path) => {
    setActiveFile(path);
    setOpenFiles(prev => prev.includes(path) ? prev : [...prev, path]);
  };

  const handleNewFile = () => {
    const path = window.prompt('Enter file path (e.g. index.html, src/app.js):');
    if (!path) return;
    if (files.some(f => f.path === path)) return;
    const newFile = { path, content: '', language: getLanguageFromPath(path) };
    setFiles(prev => [...prev, newFile]);
    openFile(path);
    broadcast({ type: 'file_create', path, file: newFile });
  };

  const handleDeleteFile = (path) => {
    if (!window.confirm(`Delete ${path}?`)) return;
    setFiles(prev => prev.filter(f => f.path !== path));
    setOpenFiles(prev => prev.filter(p => p !== path));
    setActiveFile(prev => (prev === path ? '' : prev));
    broadcast({ type: 'file_delete', path });
  };

  const handleRenameFile = (path) => {
    const next = window.prompt('Rename file to:', path);
    if (!next || next === path) return;
    if (files.some(f => f.path === next)) return;
    setFiles(prev => prev.map(f => f.path === path ? { ...f, path: next } : f));
    setOpenFiles(prev => prev.map(p => p === path ? next : p));
    setActiveFile(prev => (prev === path ? next : prev));
    broadcast({ type: 'file_rename', oldPath: path, newPath: next });
  };

  const updatePackageJson = (updater) => {
    const current = packageJsonFile?.content || '{\\n  \"name\": \"app\",\\n  \"version\": \"1.0.0\",\\n  \"dependencies\": {}\\n}';
    let json;
    try {
      json = JSON.parse(current);
    } catch (err) {
      json = { name: 'app', version: '1.0.0', dependencies: {} };
    }
    json.dependencies = json.dependencies || {};
    updater(json);
    applyFileUpdate('package.json', JSON.stringify(json, null, 2));
  };

  const handleAddNpmDependency = () => {
    const pkg = npmPackageInput.trim();
    if (!pkg) return;
    updatePackageJson((json) => {
      json.dependencies[pkg] = json.dependencies[pkg] || 'latest';
    });
    setNpmPackageInput('');
  };

  const handleRemoveNpmDependency = (pkg) => {
    if (!pkg) return;
    updatePackageJson((json) => {
      if (json.dependencies && json.dependencies[pkg]) {
        delete json.dependencies[pkg];
      }
    });
  };

  const handleAddPipDependency = () => {
    const pkg = pipPackageInput.trim();
    if (!pkg) return;
    const next = Array.from(new Set([...requirementsList, pkg]));
    applyFileUpdate('requirements.txt', next.join('\\n') + '\\n');
    setPipPackageInput('');
  };

  const handleRemovePipDependency = (pkg) => {
    const next = requirementsList.filter(item => item !== pkg);
    applyFileUpdate('requirements.txt', next.join('\\n') + (next.length ? '\\n' : ''));
  };

  const handleGenerate = async () => {
    if (!prompt.trim() || generating) return;

    const userMsg = { role: 'user', content: prompt.trim(), timestamp: new Date().toISOString() };
    setMessages(prev => [...prev, userMsg]);
    setPrompt('');
    setGenerating(true);

    try {
      const data = await api.generate({
        prompt: userMsg.content,
        currentCode: indexHtmlContent || null,
        projectId: initialProject?.id,
        provider: llmProvider || undefined,
        model: llmModel || undefined,
        stack,
        agent: activeAgent,
      });

      if (data.code) {
        if (Array.isArray(data.files) && data.files.length > 0) {
          const nextActive = data.files.some(f => f.path === activeFile)
            ? activeFile
            : (data.files.find(f => f.path === 'index.html')?.path || data.files[0].path);
          setFiles(data.files);
          setOpenFiles(prev => {
            const next = new Set(prev);
            data.files.forEach(f => next.add(f.path));
            return Array.from(next);
          });
          setActiveFile(nextActive || '');
          const nextActiveFile = data.files.find(f => f.path === nextActive);
          if (nextActiveFile) {
            setCode(nextActiveFile.content || '');
          }
        } else {
          setFiles(prev => {
            if (!prev.length) {
              return [{ path: 'index.html', content: data.code, language: 'html' }];
            }
            let found = false;
            const updated = prev.map(f => {
              if (f.path === 'index.html') {
                found = true;
                return { ...f, content: data.code, language: f.language || 'html' };
              }
              return f;
            });
            if (!found) {
              updated.push({ path: 'index.html', content: data.code, language: 'html' });
            }
            return updated;
          });
          setOpenFiles(prev => prev.includes('index.html') ? prev : [...prev, 'index.html']);
        }
        setPreviewKey(prev => prev + 1);
        const assistantMsg = {
          role: 'assistant',
          content: 'Code generated successfully!',
          timestamp: new Date().toISOString()
        };
        setMessages(prev => [...prev, assistantMsg]);
      }
    } catch (err) {
      const errorMsg = {
        role: 'assistant',
        content: `Error: ${err.message}`,
        timestamp: new Date().toISOString(),
        isError: true
      };
      setMessages(prev => [...prev, errorMsg]);
    } finally {
      setGenerating(false);
    }
  };

  const handleSave = async () => {
    if (!initialProject?.id) return;
    setSaving(true);
    try {
      await api.updateProject(initialProject.id, {
        code: indexHtmlContent || code,
        name: projectName,
        messages,
        files,
        envVars,
        settings: {
          stack,
          runProfileId,
          agent: activeAgent,
        },
      });
      setSaved(true);
      setTimeout(() => setSaved(false), 2000);
    } catch (err) {
      console.error('Save failed:', err);
    } finally {
      setSaving(false);
    }
  };

  const handleDownload = async () => {
    if (!initialProject?.id) return;
    setExporting(true);
    try {
      const blob = await api.exportProject(initialProject.id);
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${projectName.replace(/\s+/g, '-').toLowerCase()}.zip`;
      a.click();
      URL.revokeObjectURL(url);
    } catch (err) {
      console.error('Export failed:', err);
    } finally {
      setExporting(false);
    }
  };

  const handleImport = async (file) => {
    if (!initialProject?.id || !file) return;
    setImporting(true);
    try {
      const data = await api.importProject(initialProject.id, file);
      if (Array.isArray(data.files)) {
        setFiles(data.files);
        const first = data.files[0];
        setOpenFiles(first ? [first.path] : []);
        setActiveFile(first ? first.path : '');
        setCode(first ? first.content || '' : '');
        setPreviewKey(prev => prev + 1);
      }
    } catch (err) {
      console.error('Import failed:', err);
    } finally {
      setImporting(false);
    }
  };

  const handleCopyShareLink = () => {
    const baseUrl = typeof window !== 'undefined' ? window.location.origin : '';
    const apiBase = API_BASE || baseUrl;
    const shareUrl = `${apiBase}/api/preview/${initialProject?.id}`;
    navigator.clipboard.writeText(shareUrl);
    setCopied(true);
    setTimeout(() => setCopied(false), 2000);
  };

  const handleCodeChange = (value) => {
    const nextValue = value || '';
    setCode(nextValue);
    if (!activeFile) return;
    setFiles(prev => prev.map(f => {
      if (f.path === activeFile) {
        return { ...f, content: nextValue, language: f.language || getLanguageFromPath(activeFile) };
      }
      return f;
    }));
    if (broadcastTimerRef.current) {
      clearTimeout(broadcastTimerRef.current);
    }
    broadcastTimerRef.current = setTimeout(() => {
      broadcast({ type: 'file_update', path: activeFile, content: nextValue });
    }, 400);
  };

  const handleCreateSnapshot = async () => {
    if (!initialProject?.id) return;
    const name = window.prompt('Snapshot name:', `Snapshot ${new Date().toLocaleString()}`);
    if (!name) return;
    await api.createSnapshot({ projectId: initialProject.id, name });
    const data = await api.listSnapshots(initialProject.id);
    setSnapshots(data.snapshots || []);
  };

  const handleRestoreSnapshot = async (snapshotId) => {
    if (!snapshotId) return;
    if (!window.confirm('Restore this snapshot? This will overwrite current files.')) return;
    const data = await api.restoreSnapshot(snapshotId);
    if (data.project) {
      const projectFiles = data.project.files || [];
      setFiles(projectFiles);
      setOpenFiles(projectFiles.length ? [projectFiles[0].path] : []);
      setActiveFile(projectFiles.length ? projectFiles[0].path : '');
      setCode(projectFiles.length ? (projectFiles[0].content || '') : (data.project.code || ''));
      setEnvVars(data.project.envVars || []);
      setPreviewKey(prev => prev + 1);
    }
  };

  const sendTerminalCommand = (command) => {
    if (!command.trim() || !initialProject?.id) return;
    const ws = terminalSocketRef.current;
    setTerminalLogs(prev => prev + `\n$ ${command}\n`);
    if (!ws || ws.readyState !== 1) {
      terminalReconnectRef.current?.();
      setTerminalLogs(prev => prev + '[terminal disconnected] Unable to send command. Reconnect and try again.\n');
      return;
    }
    ws.send(command + '\n');
  };

  const handleTerminalRun = async () => {
    if (!terminalCommand.trim()) return;
    sendTerminalCommand(terminalCommand);
    setTerminalCommand('');
  };

  const handleRunProfile = () => {
    if (!runProfileCommand) return;
    sendTerminalCommand(runProfileCommand);
  };

  const enabledProviders = (llmOptions?.providers || []).filter(p => p.enabled);
  const activeProvider = llmProvider || llmOptions?.defaultProvider || 'groq';
  const activeModel = llmModel || llmOptions?.defaultModel || '';
  const activeModelLabel = (llmModels.find(m => m.id === activeModel) || {}).label || activeModel;
  const activeFileObj = getFileByPath(activeFile) || files[0];
  const activeFileLabel = activeFileObj?.path || 'No file';
  const filteredFiles = files
    .filter(f => f.path.toLowerCase().includes(fileSearch.toLowerCase()))
    .sort((a, b) => a.path.localeCompare(b.path));
  const activeRunProfile = runProfiles.find(p => p.id === runProfileId)
    || runProfiles.find(p => p.stack === stack)
    || runProfiles[0];
  const runProfileCommand = activeRunProfile?.command || '';
  const hasIndexHtml = files.some(f => f.path === 'index.html');

  const diffView = useMemo(() => {
    if (!diffSnapshot || !diffFilePath) return '';
    const snapshotFile = (diffSnapshot.files || []).find(f => f.path === diffFilePath);
    const currentFile = files.find(f => f.path === diffFilePath);
    const oldText = snapshotFile?.content || '';
    const newText = currentFile?.content || '';
    const oldLines = oldText.split('\n');
    const newLines = newText.split('\n');
    const maxLen = Math.max(oldLines.length, newLines.length);
    const output = [];
    for (let i = 0; i < maxLen; i += 1) {
      const oldLine = oldLines[i];
      const newLine = newLines[i];
      if (oldLine === newLine) {
        if (oldLine !== undefined) output.push(`  ${oldLine}`);
      } else {
        if (oldLine !== undefined) output.push(`- ${oldLine}`);
        if (newLine !== undefined) output.push(`+ ${newLine}`);
      }
    }
    return output.join('\n');
  }, [diffSnapshot, diffFilePath, files]);

  // Refresh preview with debounce
  useEffect(() => {
    const timer = setTimeout(() => {
      setPreviewKey(prev => prev + 1);
    }, 500);
    return () => clearTimeout(timer);
  }, [previewHtml]);

  // Keep editor content in sync with active file
  useEffect(() => {
    if (!files.length) return;
    const active = getFileByPath(activeFile) || files[0];
    if (active && active.content !== code) {
      setCode(active.content || '');
    }
  }, [activeFile, files]);

  useEffect(() => {
    if (!activeFile && files.length) {
      setActiveFile(files[0].path);
      setOpenFiles(prev => prev.length ? prev : [files[0].path]);
    }
  }, [files, activeFile]);

  return (
    <div className="h-screen flex flex-col bg-slate-950">
      {/* Top Toolbar */}
      <div className="h-12 border-b border-slate-800/50 bg-slate-950/95 backdrop-blur-xl flex items-center justify-between px-3 shrink-0">
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="icon"
            onClick={onBack}
            className="h-8 w-8 text-slate-400 hover:text-slate-100"
          >
            <ArrowLeft className="w-4 h-4" />
          </Button>
          <Separator orientation="vertical" className="h-5 bg-slate-800" />
          <div className="flex items-center gap-2">
            <div className="w-6 h-6 rounded bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center">
              <LogoMark className="text-[10px]" />
            </div>
            <input
              value={projectName}
              onChange={(e) => setProjectName(e.target.value)}
              className="bg-transparent text-sm font-medium text-slate-200 border-none outline-none focus:text-white w-40"
            />
          </div>
        </div>
        <div className="flex items-center gap-1.5">
          <Select value={stack} onValueChange={setStack}>
            <SelectTrigger className="h-8 text-xs bg-slate-900/50 border-slate-800 text-slate-200 w-[130px]">
              <SelectValue placeholder="Stack" />
            </SelectTrigger>
            <SelectContent>
              {STACK_OPTIONS.map((opt) => (
                <SelectItem key={opt.id} value={opt.id}>
                  {opt.label}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleSave}
            disabled={saving}
            className="h-8 text-xs text-slate-400 hover:text-slate-100 gap-1.5"
          >
            {saving ? <Loader2 className="w-3.5 h-3.5 animate-spin" /> : saved ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Save className="w-3.5 h-3.5" />}
            {saved ? 'Saved' : 'Save'}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCreateSnapshot}
            className="h-8 text-xs text-slate-400 hover:text-slate-100 gap-1.5"
          >
            <Copy className="w-3.5 h-3.5" />
            Snapshot
          </Button>
          <Select
            value=""
            onValueChange={(value) => handleRestoreSnapshot(value)}
            disabled={snapshotsLoading || snapshots.length === 0}
          >
            <SelectTrigger className="h-8 text-xs bg-slate-900/50 border-slate-800 text-slate-300">
              <SelectValue placeholder={snapshots.length ? 'Restore snapshot' : 'No snapshots'} />
            </SelectTrigger>
            <SelectContent>
              {snapshots.map((snap) => (
                <SelectItem key={snap.id} value={snap.id}>
                  {snap.name}
                </SelectItem>
              ))}
            </SelectContent>
          </Select>
          <Button
            variant="ghost"
            size="sm"
            onClick={handleDownload}
            className="h-8 text-xs text-slate-400 hover:text-slate-100 gap-1.5"
          >
            <FileArchive className="w-3.5 h-3.5" />
            {exporting ? 'Exporting...' : 'Export'}
          </Button>
          <Button
            variant="ghost"
            size="sm"
            onClick={() => importInputRef.current?.click()}
            className="h-8 text-xs text-slate-400 hover:text-slate-100 gap-1.5"
            disabled={importing}
          >
            <Upload className="w-3.5 h-3.5" />
            {importing ? 'Importing...' : 'Import'}
          </Button>
          <input
            ref={importInputRef}
            type="file"
            accept=".zip"
            className="hidden"
            onChange={(e) => {
              const file = e.target.files?.[0];
              if (file) handleImport(file);
              e.target.value = '';
            }}
          />
          <Button
            variant="ghost"
            size="sm"
            onClick={handleCopyShareLink}
            className="h-8 text-xs text-slate-400 hover:text-slate-100 gap-1.5"
          >
            {copied ? <Check className="w-3.5 h-3.5 text-emerald-400" /> : <Share2 className="w-3.5 h-3.5" />}
            {copied ? 'Copied!' : 'Share'}
          </Button>
          <Separator orientation="vertical" className="h-5 bg-slate-800" />
          <Button
            variant="ghost"
            size="sm"
            onClick={() => setShowPreview(!showPreview)}
            className={`h-8 text-xs gap-1.5 ${showPreview ? 'text-violet-400' : 'text-slate-400'}`}
          >
            <Eye className="w-3.5 h-3.5" />
            Preview
          </Button>
        </div>
      </div>

      {/* Main Editor Area */}
      <div className="flex-1 overflow-hidden">
        <ResizablePanelGroup direction="vertical">
          <ResizablePanel defaultSize={72} minSize={45}>
            <div className="h-full overflow-hidden">
              <ResizablePanelGroup direction="horizontal">
          {/* AI Chat Panel */}
          <ResizablePanel defaultSize={25} minSize={18} maxSize={40}>
            <div className="h-full flex flex-col bg-slate-900/50">
              {/* Chat Header */}
              <div className="h-10 border-b border-slate-800/50 flex items-center px-4 shrink-0">
                <Sparkles className="w-4 h-4 text-violet-400 mr-2" />
                <span className="text-xs font-medium text-slate-300">AI Assistant</span>
                <Badge variant="secondary" className="ml-auto text-[10px] bg-violet-500/10 text-violet-400 border-violet-500/20">
                  {providerLabel(activeProvider)} LLM
                </Badge>
              </div>

              {/* Chat Messages */}
              <ScrollArea className="flex-1 p-4">
                <div className="space-y-4">
                  {/* Welcome message */}
                  <div className="flex gap-3">
                    <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center shrink-0 mt-0.5">
                      <Wand2 className="w-3.5 h-3.5 text-white" />
                    </div>
                    <div className="text-sm text-slate-300 leading-relaxed">
                      <p className="font-medium text-violet-400 mb-1">SelfGPT Studio</p>
                      <p>Tell me what you want to build! I can create apps, websites, dashboards, and more.</p>
                    </div>
                  </div>

                  {messages.map((msg, i) => (
                    <div key={i} className={`flex gap-3 animate-fade-in ${msg.role === 'user' ? '' : ''}`}>
                      <div className={`w-7 h-7 rounded-lg flex items-center justify-center shrink-0 mt-0.5 ${
                        msg.role === 'user'
                          ? 'bg-slate-700'
                          : msg.isError
                            ? 'bg-red-500/20'
                            : 'bg-gradient-to-br from-violet-500 to-indigo-600'
                      }`}>
                        {msg.role === 'user' ? (
                          <User className="w-3.5 h-3.5 text-slate-300" />
                        ) : (
                          <Wand2 className={`w-3.5 h-3.5 ${msg.isError ? 'text-red-400' : 'text-white'}`} />
                        )}
                      </div>
                      <div className="flex-1">
                        <p className={`text-sm leading-relaxed ${
                          msg.role === 'user'
                            ? 'text-slate-200'
                            : msg.isError
                              ? 'text-red-400'
                              : 'text-emerald-400'
                        }`}>
                          {msg.content}
                        </p>
                        {msg.timestamp && (
                          <p className="text-[10px] text-slate-600 mt-1">
                            {new Date(msg.timestamp).toLocaleTimeString()}
                          </p>
                        )}
                      </div>
                    </div>
                  ))}

                  {generating && (
                    <div className="flex gap-3 animate-fade-in">
                      <div className="w-7 h-7 rounded-lg bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center shrink-0 glow-pulse">
                        <Wand2 className="w-3.5 h-3.5 text-white" />
                      </div>
                      <div className="flex items-center gap-2">
                        <Loader2 className="w-4 h-4 animate-spin text-violet-400" />
                        <span className="text-sm text-violet-400">Generating code...</span>
                      </div>
                    </div>
                  )}
                  <div ref={chatEndRef} />
                </div>
              </ScrollArea>

              {/* Chat Input */}
              <div className="p-3 border-t border-slate-800/50 shrink-0">
                <div className="flex gap-2 mb-2">
                  <Select
                    value={activeProvider}
                    onValueChange={(value) => setLlmProvider(value)}
                    disabled={llmLoading || enabledProviders.length === 0}
                  >
                    <SelectTrigger className="h-8 text-xs bg-slate-800/50 border-slate-700 text-slate-200">
                      <SelectValue placeholder="Provider" />
                    </SelectTrigger>
                    <SelectContent>
                      {enabledProviders.map((p) => (
                        <SelectItem key={p.id} value={p.id}>
                          {providerLabel(p.id)}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Select
                    value={activeModel}
                    onValueChange={(value) => setLlmModel(value)}
                    disabled={llmModelsLoading || llmModels.length === 0}
                  >
                    <SelectTrigger className="h-8 text-xs bg-slate-800/50 border-slate-700 text-slate-200">
                      <SelectValue placeholder="Model" />
                    </SelectTrigger>
                    <SelectContent>
                      {llmModels.map((m) => (
                        <SelectItem key={m.id} value={m.id}>
                          {m.label || m.id}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <div className="flex gap-2 mb-2">
                  <Select
                    value={activeAgent}
                    onValueChange={(value) => setActiveAgent(value)}
                  >
                    <SelectTrigger className="h-8 text-xs bg-slate-800/50 border-slate-700 text-slate-200 flex-1">
                      <SelectValue placeholder="Agent" />
                    </SelectTrigger>
                    <SelectContent>
                      {(agents.length ? agents : [{ id: 'builder', label: 'Builder' }]).map((agent) => (
                        <SelectItem key={agent.id} value={agent.id}>
                          {agent.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <div className="h-8 px-2 rounded-md border border-slate-800/60 bg-slate-900/40 flex items-center text-[10px] text-slate-400">
                    <Bot className="w-3 h-3 mr-1" />
                    {agents.find(a => a.id === activeAgent)?.description || 'Agent mode'}
                  </div>
                </div>
                <div className="flex gap-2">
                  <Input
                    placeholder="Describe your app..."
                    value={prompt}
                    onChange={(e) => setPrompt(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && !e.shiftKey && handleGenerate()}
                    disabled={generating}
                    className="bg-slate-800/50 border-slate-700 focus:border-violet-500 text-sm text-slate-100 placeholder:text-slate-500"
                  />
                  <Button
                    onClick={handleGenerate}
                    disabled={!prompt.trim() || generating}
                    size="icon"
                    className="bg-gradient-to-r from-violet-600 to-indigo-600 hover:from-violet-500 hover:to-indigo-500 text-white shrink-0"
                  >
                    {generating ? <Loader2 className="w-4 h-4 animate-spin" /> : <Send className="w-4 h-4" />}
                  </Button>
                </div>
                <p className="text-[10px] text-slate-600 mt-2 text-center">
                  Powered by {providerLabel(activeProvider)}{activeModelLabel ? ` + ${activeModelLabel}` : ''}
                </p>
              </div>
            </div>
          </ResizablePanel>

          <ResizableHandle withHandle />

          {/* Code Editor Panel */}
          <ResizablePanel defaultSize={showPreview ? 40 : 75} minSize={25}>
            <div className="h-full flex flex-col">
              <div className="h-10 border-b border-slate-800/50 flex items-center px-4 bg-slate-900/30 shrink-0">
                <Code2 className="w-4 h-4 text-orange-400 mr-2" />
                <span className="text-xs font-medium text-slate-300">{activeFileLabel}</span>
                <Badge variant="secondary" className="ml-2 text-[10px] bg-orange-500/10 text-orange-400 border-orange-500/20">
                  {getLanguageFromPath(activeFile).toUpperCase()}
                </Badge>
                <div className="ml-auto flex items-center gap-2">
                  <Button
                    variant="ghost"
                    size="icon"
                    onClick={handleNewFile}
                    className="h-7 w-7 text-slate-500 hover:text-slate-200"
                  >
                    <FilePlus2 className="w-3.5 h-3.5" />
                  </Button>
                </div>
              </div>
              <div className="flex-1 flex overflow-hidden">
                <div className="w-52 border-r border-slate-800/50 bg-slate-900/40 flex flex-col">
                  <div className="p-2 border-b border-slate-800/50">
                    <div className="flex items-center gap-2">
                      <Search className="w-3.5 h-3.5 text-slate-500" />
                      <Input
                        placeholder="Search files"
                        value={fileSearch}
                        onChange={(e) => setFileSearch(e.target.value)}
                        className="h-7 text-xs bg-slate-800/50 border-slate-700 text-slate-100 placeholder:text-slate-500"
                      />
                    </div>
                  </div>
                  <ScrollArea className="flex-1">
                    <div className="p-2 space-y-1">
                      {filteredFiles.map((file) => {
                        const depth = file.path.split('/').length - 1;
                        return (
                          <div
                            key={file.path}
                            className={`group flex items-center gap-2 px-2 py-1 rounded text-xs cursor-pointer ${
                              activeFile === file.path ? 'bg-slate-800/60 text-slate-100' : 'text-slate-400 hover:bg-slate-800/40'
                            }`}
                            style={{ paddingLeft: 8 + depth * 10 }}
                            onClick={() => openFile(file.path)}
                          >
                            <FolderOpen className="w-3.5 h-3.5 text-slate-500" />
                            <span className="truncate flex-1">{file.path.split('/').pop()}</span>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={(e) => { e.stopPropagation(); handleRenameFile(file.path); }}
                              className="h-6 w-6 text-slate-500 hover:text-slate-200 opacity-0 group-hover:opacity-100"
                            >
                              <Pencil className="w-3 h-3" />
                            </Button>
                            <Button
                              variant="ghost"
                              size="icon"
                              onClick={(e) => { e.stopPropagation(); handleDeleteFile(file.path); }}
                              className="h-6 w-6 text-slate-500 hover:text-red-400 opacity-0 group-hover:opacity-100"
                            >
                              <Trash2 className="w-3 h-3" />
                            </Button>
                          </div>
                        );
                      })}
                      {filteredFiles.length === 0 && (
                        <p className="text-xs text-slate-600 px-2 py-2">No files found</p>
                      )}
                    </div>
                  </ScrollArea>
                </div>
                <div className="flex-1 flex flex-col">
                  <div className="h-8 border-b border-slate-800/50 flex items-center gap-2 px-2 bg-slate-900/30">
                    {openFiles.map((path) => (
                      <div
                        key={path}
                        className={`flex items-center gap-2 px-2 py-1 rounded text-xs cursor-pointer ${
                          activeFile === path ? 'bg-slate-800/60 text-slate-100' : 'text-slate-400 hover:bg-slate-800/40'
                        }`}
                        onClick={() => setActiveFile(path)}
                      >
                        <span className="truncate max-w-[140px]">{path.split('/').pop()}</span>
                        <button
                          onClick={(e) => {
                            e.stopPropagation();
                            setOpenFiles(prev => prev.filter(p => p !== path));
                            if (activeFile === path) {
                              const remaining = openFiles.filter(p => p !== path);
                              setActiveFile(remaining[0] || '');
                            }
                          }}
                          className="text-slate-500 hover:text-slate-200"
                        >
                          <X className="w-3 h-3" />
                        </button>
                      </div>
                    ))}
                  </div>
                  <div className="flex-1 editor-container">
                    <MonacoEditor
                      height="100%"
                      language={getLanguageFromPath(activeFile)}
                      theme="vs-dark"
                      value={code}
                      onChange={handleCodeChange}
                      options={{
                        minimap: { enabled: false },
                        fontSize: 13,
                        lineHeight: 20,
                        wordWrap: 'on',
                        lineNumbers: 'on',
                        scrollBeyondLastLine: false,
                        automaticLayout: true,
                        padding: { top: 8 },
                        renderWhitespace: 'none',
                        smoothScrolling: true,
                        cursorSmoothCaretAnimation: 'on',
                        bracketPairColorization: { enabled: true },
                      }}
                    />
                  </div>
                </div>
              </div>
            </div>
          </ResizablePanel>

          {showPreview && (
            <>
              <ResizableHandle withHandle />

              {/* Preview Panel */}
              <ResizablePanel defaultSize={35} minSize={20}>
                <div className="h-full flex flex-col">
                  <div className="h-10 border-b border-slate-800/50 flex items-center px-4 bg-slate-900/30 shrink-0">
                    <Eye className="w-4 h-4 text-emerald-400 mr-2" />
                    <span className="text-xs font-medium text-slate-300">Live Preview</span>
                    <div className="ml-auto flex items-center gap-1.5">
                      <Button
                        variant="ghost"
                        size="icon"
                        onClick={() => setPreviewKey(prev => prev + 1)}
                        className="h-6 w-6 text-slate-500 hover:text-slate-300"
                      >
                        <RefreshCw className="w-3 h-3" />
                      </Button>
                      {initialProject?.id && (
                        <Button
                          variant="ghost"
                          size="icon"
                          onClick={() => {
                            const baseUrl = typeof window !== 'undefined' ? window.location.origin : '';
                            const apiBase = API_BASE || baseUrl;
                            window.open(`${apiBase}/api/preview/${initialProject.id}`, '_blank');
                          }}
                          className="h-6 w-6 text-slate-500 hover:text-slate-300"
                        >
                          <ExternalLink className="w-3 h-3" />
                        </Button>
                      )}
                    </div>
                  </div>
                  <div className="flex-1 bg-white relative">
                    {!hasIndexHtml && (
                      <div className="absolute inset-0 flex flex-col items-center justify-center text-center text-slate-700 bg-slate-50/90 z-10 p-6">
                        <div className="text-lg font-semibold mb-2">No HTML preview</div>
                        <div className="text-sm text-slate-500 mb-4">
                          This looks like a backend or framework project. Run it from Terminal to see output.
                        </div>
                        {runProfileCommand && (
                          <div className="text-xs text-slate-600 bg-slate-100 border border-slate-200 rounded px-3 py-2">
                            Suggested: <span className="font-mono">{runProfileCommand}</span>
                          </div>
                        )}
                      </div>
                    )}
                    <iframe
                      key={previewKey}
                      srcDoc={previewHtml}
                      className="w-full h-full border-0"
                      sandbox="allow-scripts allow-same-origin"
                      title="Preview"
                    />
                  </div>
                </div>
              </ResizablePanel>
            </>
          )}
          </ResizablePanelGroup>
            </div>
          </ResizablePanel>

          <ResizableHandle withHandle className="bg-slate-800/50" />

          {/* Bottom Panel */}
          <ResizablePanel defaultSize={28} minSize={18} maxSize={45}>
            <div className="h-full border-t border-slate-800/50 bg-slate-950/80">
              <Tabs value={bottomTab} onValueChange={setBottomTab} className="h-full">
            <div className="h-9 border-b border-slate-800/50 flex items-center px-3">
              <TabsList className="h-7 bg-slate-900/60">
                <TabsTrigger value="terminal" className="text-[11px]">Terminal</TabsTrigger>
                <TabsTrigger value="deps" className="text-[11px]">Deps</TabsTrigger>
                <TabsTrigger value="diff" className="text-[11px]">Diff</TabsTrigger>
                <TabsTrigger value="insights" className="text-[11px]">Insights</TabsTrigger>
                <TabsTrigger value="env" className="text-[11px]">Env</TabsTrigger>
              </TabsList>
              <div className="ml-auto flex items-center gap-2">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setBottomTab('terminal')}
                  className="h-7 w-7 text-slate-500 hover:text-slate-200"
                >
                  <Terminal className="w-3.5 h-3.5" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setBottomTab('deps')}
                  className="h-7 w-7 text-slate-500 hover:text-slate-200"
                >
                  <Package className="w-3.5 h-3.5" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setBottomTab('diff')}
                  className="h-7 w-7 text-slate-500 hover:text-slate-200"
                >
                  <GitCompare className="w-3.5 h-3.5" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setBottomTab('insights')}
                  className="h-7 w-7 text-slate-500 hover:text-slate-200"
                >
                  <Zap className="w-3.5 h-3.5" />
                </Button>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setBottomTab('env')}
                  className="h-7 w-7 text-slate-500 hover:text-slate-200"
                >
                  <Settings className="w-3.5 h-3.5" />
                </Button>
              </div>
            </div>

            <TabsContent value="terminal" className="h-[calc(100%-36px)] m-0">
              <div className="h-full flex flex-col">
                <div className="p-2 flex items-center gap-2 border-b border-slate-800/50">
                  <Select value={runProfileId} onValueChange={setRunProfileId}>
                    <SelectTrigger className="h-7 text-xs bg-slate-800/50 border-slate-700 text-slate-200 w-60">
                      <SelectValue placeholder="Run profile" />
                    </SelectTrigger>
                    <SelectContent>
                      {runProfiles.map((profile) => (
                        <SelectItem key={profile.id} value={profile.id}>
                          {profile.label}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Button
                    size="sm"
                    onClick={handleRunProfile}
                    disabled={!runProfileCommand || terminalStatus !== 'connected'}
                    className="h-7 text-xs bg-emerald-600 hover:bg-emerald-500"
                  >
                    Run Profile
                  </Button>
                  {runProfileCommand && (
                    <span className="text-[10px] text-slate-500 truncate">{runProfileCommand}</span>
                  )}
                </div>
                <div className="p-2 flex items-center gap-2 border-b border-slate-800/50">
                  <Input
                    placeholder="Enter command (e.g. ls, python --version)"
                    value={terminalCommand}
                    onChange={(e) => setTerminalCommand(e.target.value)}
                    onKeyDown={(e) => e.key === 'Enter' && handleTerminalRun()}
                    className="h-7 text-xs bg-slate-800/50 border-slate-700 text-slate-100 placeholder:text-slate-500 flex-1"
                  />
                  <Button
                    size="sm"
                    onClick={handleTerminalRun}
                    disabled={terminalStatus !== 'connected'}
                    className="h-7 text-xs bg-violet-600 hover:bg-violet-500"
                  >
                    <Terminal className="w-3.5 h-3.5 mr-1" />
                    Send
                  </Button>
                </div>
                <div className="px-2 py-1 flex items-center gap-2 text-[10px] text-slate-400 border-b border-slate-800/50">
                  <span
                    className={`h-2 w-2 rounded-full ${
                      terminalStatus === 'connected'
                        ? 'bg-emerald-400'
                        : terminalStatus === 'connecting'
                        ? 'bg-amber-400'
                        : 'bg-rose-400'
                    }`}
                  />
                  <span className="uppercase tracking-wide">{terminalStatus}</span>
                  {terminalStatusDetail && (
                    <span className="text-slate-500">· {terminalStatusDetail}</span>
                  )}
                  {terminalStatus !== 'connected' && (
                    <Button
                      size="sm"
                      variant="ghost"
                      onClick={() => terminalReconnectRef.current?.()}
                      className="h-6 text-[10px] text-slate-400 hover:text-slate-100"
                    >
                      Reconnect
                    </Button>
                  )}
                </div>
                <ScrollArea className="flex-1 p-2">
                  <pre className="text-xs text-slate-300 whitespace-pre-wrap">{terminalLogs || 'Terminal output will appear here.'}</pre>
                </ScrollArea>
              </div>
            </TabsContent>

            <TabsContent value="deps" className="h-[calc(100%-36px)] m-0">
              <div className="h-full flex flex-col">
                <div className="p-2 border-b border-slate-800/50 flex items-center gap-2 text-xs text-slate-400">
                  <Package className="w-3.5 h-3.5" />
                  Manage dependencies directly in files.
                </div>
                <ScrollArea className="flex-1 p-3">
                  <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
                    <div className="rounded-lg border border-slate-800/60 p-3 bg-slate-900/40">
                      <div className="text-xs font-semibold text-slate-200 mb-2">Node (package.json)</div>
                      <div className="flex gap-2 mb-2">
                        <Input
                          placeholder="Add package (e.g. express)"
                          value={npmPackageInput}
                          onChange={(e) => setNpmPackageInput(e.target.value)}
                          className="h-7 text-xs bg-slate-800/50 border-slate-700 text-slate-100 placeholder:text-slate-500"
                        />
                        <Button
                          size="sm"
                          onClick={handleAddNpmDependency}
                          className="h-7 text-xs bg-emerald-600 hover:bg-emerald-500"
                        >
                          Add
                        </Button>
                      </div>
                      <div className="space-y-1 text-xs text-slate-400">
                        {Object.keys(packageDeps).length === 0 && (
                          <div>No dependencies found.</div>
                        )}
                        {Object.entries(packageDeps).map(([name, version]) => (
                          <div key={name} className="flex items-center justify-between">
                            <span>{name} <span className="text-slate-500">{version}</span></span>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => handleRemoveNpmDependency(name)}
                              className="h-6 text-[10px] text-slate-400 hover:text-red-400"
                            >
                              remove
                            </Button>
                          </div>
                        ))}
                      </div>
                    </div>

                    <div className="rounded-lg border border-slate-800/60 p-3 bg-slate-900/40">
                      <div className="text-xs font-semibold text-slate-200 mb-2">Python (requirements.txt)</div>
                      <div className="flex gap-2 mb-2">
                        <Input
                          placeholder="Add package (e.g. fastapi)"
                          value={pipPackageInput}
                          onChange={(e) => setPipPackageInput(e.target.value)}
                          className="h-7 text-xs bg-slate-800/50 border-slate-700 text-slate-100 placeholder:text-slate-500"
                        />
                        <Button
                          size="sm"
                          onClick={handleAddPipDependency}
                          className="h-7 text-xs bg-emerald-600 hover:bg-emerald-500"
                        >
                          Add
                        </Button>
                      </div>
                      <div className="space-y-1 text-xs text-slate-400">
                        {requirementsList.length === 0 && (
                          <div>No requirements found.</div>
                        )}
                        {requirementsList.map((name) => (
                          <div key={name} className="flex items-center justify-between">
                            <span>{name}</span>
                            <Button
                              size="sm"
                              variant="ghost"
                              onClick={() => handleRemovePipDependency(name)}
                              className="h-6 text-[10px] text-slate-400 hover:text-red-400"
                            >
                              remove
                            </Button>
                          </div>
                        ))}
                      </div>
                    </div>
                  </div>
                </ScrollArea>
              </div>
            </TabsContent>

            <TabsContent value="diff" className="h-[calc(100%-36px)] m-0">
              <div className="h-full flex flex-col">
                <div className="p-2 border-b border-slate-800/50 flex items-center gap-2">
                  <Select
                    value={diffSnapshotId}
                    onValueChange={(value) => setDiffSnapshotId(value)}
                  >
                    <SelectTrigger className="h-7 text-xs bg-slate-800/50 border-slate-700 text-slate-200 w-56">
                      <SelectValue placeholder="Select snapshot" />
                    </SelectTrigger>
                    <SelectContent>
                      {snapshots.map((snap) => (
                        <SelectItem key={snap.id} value={snap.id}>
                          {snap.name}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                  <Select
                    value={diffFilePath}
                    onValueChange={(value) => setDiffFilePath(value)}
                    disabled={!diffSnapshot}
                  >
                    <SelectTrigger className="h-7 text-xs bg-slate-800/50 border-slate-700 text-slate-200 w-56">
                      <SelectValue placeholder="Select file" />
                    </SelectTrigger>
                    <SelectContent>
                      {(diffSnapshot?.files || []).map((f) => (
                        <SelectItem key={f.path} value={f.path}>
                          {f.path}
                        </SelectItem>
                      ))}
                    </SelectContent>
                  </Select>
                </div>
                <ScrollArea className="flex-1 p-2">
                  <pre className="text-xs text-slate-300 whitespace-pre-wrap">{diffView || 'Select a snapshot and file to view diff.'}</pre>
                </ScrollArea>
              </div>
            </TabsContent>

            <TabsContent value="insights" className="h-[calc(100%-36px)] m-0">
              <div className="h-full flex flex-col">
                <div className="p-2 border-b border-slate-800/50 flex items-center gap-2 text-xs text-slate-400">
                  <Zap className="w-3.5 h-3.5" />
                  Parsed errors and suggested fixes.
                </div>
                <ScrollArea className="flex-1 p-3">
                  <div className="space-y-2 text-xs text-slate-300">
                    {insights.length === 0 && (
                      <div className="text-slate-500">No insights yet. Run commands in Terminal to generate logs.</div>
                    )}
                    {insights.map((item, idx) => (
                      <div key={idx} className="rounded-lg border border-slate-800/60 p-3 bg-slate-900/40">
                        <div className="font-semibold text-slate-200">{item.title}</div>
                        <div className="text-slate-400">{item.detail}</div>
                        {item.suggestion && (
                          <div className="text-emerald-400 mt-1">Suggestion: {item.suggestion}</div>
                        )}
                      </div>
                    ))}
                  </div>
                </ScrollArea>
              </div>
            </TabsContent>

            <TabsContent value="env" className="h-[calc(100%-36px)] m-0">
              <div className="h-full flex flex-col">
                <div className="p-2 flex items-center gap-2 border-b border-slate-800/50">
                  <Button
                    size="sm"
                    onClick={() => setEnvVars(prev => [...prev, { key: '', value: '', secret: false }])}
                    className="h-7 text-xs bg-slate-800/50 hover:bg-slate-700"
                  >
                    <Plus className="w-3.5 h-3.5 mr-1" />
                    Add Var
                  </Button>
                  <Button
                    size="sm"
                    variant="ghost"
                    onClick={handleSave}
                    className="h-7 text-xs text-slate-400 hover:text-slate-100"
                  >
                    Save Env
                  </Button>
                </div>
                <ScrollArea className="flex-1 p-2">
                  <div className="space-y-2">
                    {envVars.map((item, idx) => (
                      <div key={idx} className="flex items-center gap-2">
                        <Input
                          placeholder="KEY"
                          value={item.key}
                          onChange={(e) => {
                            const value = e.target.value;
                            setEnvVars(prev => prev.map((v, i) => i === idx ? { ...v, key: value } : v));
                          }}
                          className="h-7 text-xs bg-slate-800/50 border-slate-700 text-slate-100 placeholder:text-slate-500 w-48"
                        />
                        <Input
                          type={item.secret ? 'password' : 'text'}
                          placeholder="VALUE"
                          value={item.value}
                          onChange={(e) => {
                            const value = e.target.value;
                            setEnvVars(prev => prev.map((v, i) => i === idx ? { ...v, value } : v));
                          }}
                          className="h-7 text-xs bg-slate-800/50 border-slate-700 text-slate-100 placeholder:text-slate-500 flex-1"
                        />
                        <label className="flex items-center gap-1 text-[10px] text-slate-400">
                          <input
                            type="checkbox"
                            checked={!!item.secret}
                            onChange={(e) => {
                              const value = e.target.checked;
                              setEnvVars(prev => prev.map((v, i) => i === idx ? { ...v, secret: value } : v));
                            }}
                          />
                          secret
                        </label>
                        <Button
                          size="icon"
                          variant="ghost"
                          onClick={() => setEnvVars(prev => prev.filter((_, i) => i !== idx))}
                          className="h-7 w-7 text-slate-500 hover:text-red-400"
                        >
                          <Trash2 className="w-3.5 h-3.5" />
                        </Button>
                      </div>
                    ))}
                    {envVars.length === 0 && (
                      <p className="text-xs text-slate-600">No environment variables yet.</p>
                    )}
                  </div>
                </ScrollArea>
              </div>
            </TabsContent>
              </Tabs>
            </div>
          </ResizablePanel>
        </ResizablePanelGroup>
      </div>
    </div>
  );
}

// --- Main App ---
export default function App() {
  const [view, setView] = useState('loading');
  const [user, setUser] = useState(null);
  const [currentProject, setCurrentProject] = useState(null);

  useEffect(() => {
    const token = localStorage.getItem('token');
    const urlParams = typeof window !== 'undefined' ? new URLSearchParams(window.location.search) : null;
    const projectIdFromUrl = urlParams ? urlParams.get('projectId') : null;
    const projectIdFromStorage = localStorage.getItem('lastProjectId');
    const projectId = projectIdFromUrl || projectIdFromStorage;

    const load = async () => {
      if (!token) {
        setView('auth');
        return;
      }
      try {
        const me = await api.getMe();
        setUser(me.user);
        if (projectId) {
          const data = await api.getProject(projectId);
          if (data?.project) {
            setCurrentProject(data.project);
            setView('editor');
            return;
          }
        }
        setView('dashboard');
      } catch {
        localStorage.removeItem('token');
        setView('auth');
      }
    };

    load();
  }, []);

  const handleAuth = (userData) => {
    setUser(userData);
    const projectId = typeof window !== 'undefined' ? (new URLSearchParams(window.location.search).get('projectId') || localStorage.getItem('lastProjectId')) : null;
    if (projectId) {
      api.getProject(projectId)
        .then((data) => {
          if (data?.project) {
            setCurrentProject(data.project);
            setView('editor');
          } else {
            setView('dashboard');
          }
        })
        .catch(() => setView('dashboard'));
    } else {
      setView('dashboard');
    }
  };

  const handleLogout = () => {
    localStorage.removeItem('token');
    localStorage.removeItem('lastProjectId');
    setUser(null);
    setView('auth');
  };

  const handleOpenProject = (project) => {
    setCurrentProject(project);
    setView('editor');
    if (typeof window !== 'undefined') {
      localStorage.setItem('lastProjectId', project.id);
      const url = new URL(window.location.href);
      url.searchParams.set('projectId', project.id);
      window.history.replaceState({}, '', url.toString());
    }
  };

  const handleBackToDashboard = () => {
    setCurrentProject(null);
    setView('dashboard');
    if (typeof window !== 'undefined') {
      localStorage.removeItem('lastProjectId');
      const url = new URL(window.location.href);
      url.searchParams.delete('projectId');
      window.history.replaceState({}, '', url.toString());
    }
  };

  if (view === 'loading') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-slate-950">
        <div className="text-center">
          <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-violet-500 to-indigo-600 flex items-center justify-center mx-auto mb-4 glow-pulse">
            <LogoMark className="text-xl" />
          </div>
          <Loader2 className="w-6 h-6 animate-spin text-violet-500 mx-auto" />
        </div>
      </div>
    );
  }

  if (view === 'auth') {
    return <AuthView onAuth={handleAuth} />;
  }

  if (view === 'editor' && currentProject) {
    return <EditorView project={currentProject} onBack={handleBackToDashboard} />;
  }

  return (
    <DashboardView
      user={user}
      onLogout={handleLogout}
      onOpenProject={handleOpenProject}
    />
  );
}
