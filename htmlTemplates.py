css = '''
<style>
@import url('https://fonts.googleapis.com/css2?family=Poppins:wght@400;500;600&display=swap');

body {
    font-family: 'Poppins', sans-serif;
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    margin: 0;
    padding: 0;
}

.chat-message {
    padding: 1.5rem;
    border-radius: 1rem;
    margin-bottom: 1.5rem;
    display: flex;
    align-items: center;
    backdrop-filter: blur(12px);
    background: rgba(255, 255, 255, 0.05);
    box-shadow: 0 0 15px rgba(0, 255, 255, 0.3), 0 0 25px rgba(0, 255, 255, 0.2);
    border: 2px solid rgba(0, 255, 255, 0.4);
    transition: transform 0.3s ease, background 0.3s ease, box-shadow 0.3s ease;
}

.chat-message:hover {
    transform: scale(1.02);
    background: rgba(255, 255, 255, 0.08);
    box-shadow: 0 0 20px rgba(0, 255, 255, 0.6), 0 0 40px rgba(0, 255, 255, 0.4);
}

.chat-message.user {
    background: linear-gradient(to right, #2b313e, #1f252e);
}

.chat-message.bot {
    background: linear-gradient(to right, #475063, #3a3f4b);
}

.chat-message .avatar {
    width: 64px;
    height: 64px;
    margin-right: 1.5rem;
    flex-shrink: 0;
}

.chat-message .avatar img {
    width: 64px;
    height: 64px;
    border-radius: 50%;
    object-fit: cover;
    border: 2px solid #00ffe7;
    box-shadow: 0 0 15px rgba(0, 255, 231, 0.6);
}

.chat-message .message {
    flex: 1;
    padding: 0.5rem 1rem;
    font-size: 1rem;
    color: #f5f5f5;
    line-height: 1.6;
    word-wrap: break-word;
    text-shadow: 0 1px 2px rgba(0,0,0,0.3);
}
</style>
'''

bot_template = '''
<div class="chat-message bot">
    <div class="avatar">
        <img src="https://api.dicebear.com/6.x/bottts/svg?seed=bot" width="64" height="64" alt="Bot Avatar"/>
    </div>
    <div class="message">{{MSG}}</div>
</div>
'''

user_template = '''
<div class="chat-message user">
    <div class="avatar">
        <img src="https://api.dicebear.com/6.x/micah/svg?seed=user" width="64" height="64" alt="User Avatar"/>
    </div>    
    <div class="message">{{MSG}}</div>
</div>
'''
