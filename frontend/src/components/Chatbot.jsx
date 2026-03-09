import React, { useState, useRef, useEffect } from 'react';
import { MessageSquare, X, Send, Cpu, Bot, User } from 'lucide-react';
import { chatWithAi } from '../services/optibatchApi';


const Chatbot = () => {
  const [isOpen, setIsOpen] = useState(false);
  const [messages, setMessages] = useState([
    { role: 'assistant', text: 'OptiBatch AI Online. How can I assist with your telemetry data today?' }
  ]);
  const [inputMessage, setInputMessage] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const messagesEndRef = useRef(null);

  const scrollToBottom = () => messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  useEffect(() => { scrollToBottom(); }, [messages]);

  const handleSendMessage = async (e) => {
    e.preventDefault();
    if (!inputMessage.trim()) return;
    const userText = inputMessage.trim();
    setMessages(prev => [...prev, { role: 'user', text: userText }]);
    setInputMessage('');
    setIsLoading(true);
    try {
      const data = await chatWithAi(userText);
      if (!data) return;

      setMessages(prev => [...prev, { role: 'assistant', text: data.response || 'Communication offline.' }]);
    } catch {
      setMessages(prev => [...prev, { role: 'assistant', text: 'Error connecting to the AI Subsystem.' }]);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <>
      {!isOpen && (
        <button
          onClick={() => setIsOpen(true)}
          className="fixed bottom-8 right-8 p-4 rounded-full bg-cyan-500/10 border border-cyan-400/30 text-cyan-400 hover:bg-cyan-400/20 hover:scale-110 hover:shadow-[0_0_20px_rgba(34,211,238,0.4)] transition-all duration-300 z-50 group backdrop-blur-md"
        >
          <div className="absolute inset-0 bg-cyan-400/20 rounded-full blur-md opacity-0 group-hover:opacity-100 transition duration-300" />
          <MessageSquare size={26} className="relative z-10" />
        </button>
      )}
      {isOpen && (
        <div className="fixed bottom-8 right-8 w-full max-w-sm sm:max-w-md h-[500px] max-h-[80vh] bg-slate-900/80 backdrop-blur-xl border border-white/10 rounded-2xl shadow-[0_0_40px_rgba(0,0,0,0.5)] flex flex-col overflow-hidden z-50">
          <div className="px-5 py-4 border-b border-white/10 bg-slate-900/50 flex justify-between items-center">
            <div className="flex items-center gap-3">
              <div className="flex items-center justify-center p-2 rounded-lg bg-cyan-900/50 border border-cyan-500/30">
                <Cpu size={18} className="text-cyan-400 animate-pulse" />
              </div>
              <div>
                <h3 className="font-black text-white tracking-tight">OPTI<span className="text-cyan-400">AI</span></h3>
                <div className="flex items-center gap-2 mt-0.5">
                  <div className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
                  <span className="text-[9px] uppercase tracking-widest text-emerald-500 font-bold">System Online</span>
                </div>
              </div>
            </div>
            <button onClick={() => setIsOpen(false)} className="p-1.5 rounded-md hover:bg-white/10 text-slate-400 hover:text-white transition-colors">
              <X size={20} />
            </button>
          </div>
          <div className="flex-1 p-5 overflow-y-auto flex flex-col gap-4">
            {messages.map((msg, idx) => (
              <div key={idx} className={`flex w-full ${msg.role === 'user' ? 'justify-end' : 'justify-start'}`}>
                <div className={`flex gap-3 max-w-[85%] ${msg.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                  <div className={`flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-lg border ${msg.role === 'user' ? 'bg-slate-800 border-slate-700 text-slate-300' : 'bg-cyan-900/30 border-cyan-500/20 text-cyan-400'}`}>
                    {msg.role === 'user' ? <User size={16} /> : <Bot size={18} />}
                  </div>
                  <div className={`p-3 rounded-2xl text-[13px] leading-relaxed backdrop-blur-md ${msg.role === 'user' ? 'bg-slate-800/80 text-white rounded-tr-sm border border-slate-700' : 'bg-cyan-950/20 text-slate-200 rounded-tl-sm border border-cyan-500/10'}`}>
                    {msg.text}
                  </div>
                </div>
              </div>
            ))}
            {isLoading && (
              <div className="flex w-full justify-start">
                <div className="flex gap-3 max-w-[85%]">
                  <div className="flex-shrink-0 w-8 h-8 flex items-center justify-center rounded-lg border bg-cyan-900/30 border-cyan-500/20 text-cyan-400">
                    <Bot size={18} />
                  </div>
                  <div className="p-3 rounded-2xl text-[13px] bg-cyan-950/20 border border-cyan-500/10 flex items-center gap-1 min-w-[60px]">
                    <div className="w-1.5 h-1.5 bg-cyan-500 rounded-full animate-bounce [animation-delay:-0.3s]" />
                    <div className="w-1.5 h-1.5 bg-cyan-500 rounded-full animate-bounce [animation-delay:-0.15s]" />
                    <div className="w-1.5 h-1.5 bg-cyan-500 rounded-full animate-bounce" />
                  </div>
                </div>
              </div>
            )}
            <div ref={messagesEndRef} />
          </div>
          <div className="p-4 border-t border-white/10 bg-slate-900/80 backdrop-blur-md">
            <form onSubmit={handleSendMessage} className="relative flex items-center">
              <input
                type="text"
                value={inputMessage}
                onChange={(e) => setInputMessage(e.target.value)}
                placeholder="Query system protocols..."
                className="w-full bg-slate-950/50 text-white text-sm border border-slate-700 rounded-xl pl-4 pr-12 py-3 focus:outline-none focus:border-cyan-500/50 focus:ring-1 focus:ring-cyan-500/50 transition-all placeholder:text-slate-500"
              />
              <button
                type="submit"
                disabled={!inputMessage.trim() || isLoading}
                className="absolute right-2 p-2 rounded-lg bg-cyan-500/10 text-cyan-400 hover:bg-cyan-400 hover:text-slate-900 disabled:opacity-50 transition-colors"
              >
                <Send size={16} className={isLoading ? 'animate-pulse' : ''} />
              </button>
            </form>
          </div>
        </div>
      )}
    </>
  );
};

export default Chatbot;
