import { Chat } from "./components/Chat";
import "./App.css";

export default function App() {
  return (
    <main className="app">
      <header className="header">
        <h1>AI System Design Coach</h1>
        <p>Grounded, cited answers for production AI engineering. Every answer is measured.</p>
      </header>
      <Chat />
    </main>
  );
}
