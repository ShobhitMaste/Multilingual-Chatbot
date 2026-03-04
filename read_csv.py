import os
f = open(os.path.join("resources", "22_Multilingual_Chatbot.csv"), "r", encoding="utf-8-sig")
content = f.read(5000)
f.close()
print(repr(content[:200]))
print("---")
print(content[:3000])
print("---")
# Also get total line count
f2 = open(os.path.join("resources", "22_Multilingual_Chatbot.csv"), "r", encoding="utf-8-sig")
lines = f2.readlines()
f2.close()
print(f"Total lines: {len(lines)}")
print(f"First line (header): {lines[0].strip()}")
