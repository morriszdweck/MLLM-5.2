#!/usr/bin/env python3
"""
MLLM-5.2 — Document Autocomplete Micro Language Model
Single-file, zero-dependency, pure Python 3.10+

A tiny autocomplete LM that continues your document.
Core idea: build a causal n-gram topology from a corpus, then complete
prefixes left-to-right. Each position is scored only on left context
(distances 1..3), sampled under temperature, gated by confidence — like
ghost-text in an editor. Tab to accept.

Usage
-----
  python MLLM-5.2.py "the quick brown"                # one-shot autocomplete (default)
  python MLLM-5.2.py autocomplete "hello world" --steps 16 --seed 42
  python MLLM-5.2.py generate "what is an atom" --steps 30 --seed 42  # alias
  python MLLM-5.2.py                                   # interactive autocomplete REPL
  python MLLM-5.2.py chat --steps 24                   # alias for REPL

Principles (autocomplete-focused)
----------------------------------
  • causal n-gram topology (left context only, n=1..3)
  • left-to-right generation: prefix frozen, continuation sampled step-by-step
  • temperature + threshold gating (ghost only if confident)
  • diffusion heritage: annealed steps still available via --steps/effort
  • pure stdlib, deterministic with --seed, works offline

No install needed. Just Python 3.10+. Optionally install for `mllm52` CLI
via `pip install -e .` (package re-exports this file).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import re
import sys
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

__version__ = "5.2"
MASK = "<mask>"

# ───────────────────────────────────────────────────────────── tokenization
TOKEN_RE = re.compile(r"\b[a-zA-Z0-9']+\b|[.!?]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_RE = re.compile(r"\s+")

def tokenize(text: str) -> list[str]:
    """Split *text* into lowercase word and punctuation tokens."""
    return TOKEN_RE.findall(text.lower())

# ───────────────────────────────────────────────────────────── embedded corpus — place to define (autocomplete-based)
# Define your corpus below between the triple quotes. Replace the placeholder.
BUILT_IN_CORPUS = r"""
#!/usr/bin/env python3
"""
MLLM-5.2 — Document Autocomplete Micro Language Model
Single-file, zero-dependency, pure Python 3.10+

A tiny autocomplete LM that continues your document.
Core idea: build a causal n-gram topology from a corpus, then complete
prefixes left-to-right. Each position is scored only on left context
(distances 1..3), sampled under temperature, gated by confidence — like
ghost-text in an editor. Tab to accept.

Usage
-----
  python MLLM-5.2.py "the quick brown"                # one-shot autocomplete (default)
  python MLLM-5.2.py autocomplete "hello world" --steps 16 --seed 42
  python MLLM-5.2.py generate "what is an atom" --steps 30 --seed 42  # alias
  python MLLM-5.2.py                                   # interactive autocomplete REPL
  python MLLM-5.2.py chat --steps 24                   # alias for REPL

Principles (autocomplete-focused)
----------------------------------
  • causal n-gram topology (left context only, n=1..3)
  • left-to-right generation: prefix frozen, continuation sampled step-by-step
  • temperature + threshold gating (ghost only if confident)
  • diffusion heritage: annealed steps still available via --steps/effort
  • pure stdlib, deterministic with --seed, works offline

No install needed. Just Python 3.10+. Optionally install for `mllm52` CLI
via `pip install -e .` (package re-exports this file).
"""

from __future__ import annotations

import argparse
import logging
import math
import os
import random
import re
import sys
import textwrap
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

__version__ = "5.2"
MASK = "<mask>"

# ───────────────────────────────────────────────────────────── tokenization
TOKEN_RE = re.compile(r"\b[a-zA-Z0-9']+\b|[.!?]")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
WHITESPACE_RE = re.compile(r"\s+")

def tokenize(text: str) -> list[str]:
    """Split *text* into lowercase word and punctuation tokens."""
    return TOKEN_RE.findall(text.lower())

# ───────────────────────────────────────────────────────────── embedded corpus — place to define (autocomplete-based)
# Define your corpus below between the triple quotes. Replace the placeholder.
BUILT_IN_CORPUS = r"""
Hello, what’s up? 

Hi, how’s it doing? 

What can you do I can do many things, such as basic reasoning, text generation, etc. 

Tell me a joke Why don’t scientists trust atoms, because they make up everything! 

You are smart too, thanks for saying that! 

2+2 is equal to 4. 

1+1 is equal to 2 

3+3 is equal to 6 

4x3 is equal to 12 

Do you know math, because I don't know it so well. 

Thank you- Have a great day! 

What is your name? 

What are atoms- Atoms are the basic particles of the chemical elements and the fundamental building blocks of matter. 

What is an atom- Atoms are the basic particles of the chemical elements and the fundamental building blocks of matter. 

I’m just here to help out 

Hello, how can I help you today. 

Good morning, I hope your day is going well. 

Good afternoon, what would you like to learn. 

Good evening, I am ready to assist you. 

Thank you for your question, I will try to help clearly. 

Hi there, how can I help you? 

Hello there! How are you doing today? I hope everything is going well for you. 

 I am here to assist you with anything you need. 

Welcome to our conversation space where we can talk about many different topics. 

What would you like to discuss with me right now? I am ready to listen and respond. 

Coding is a wonderful skill that opens up many creative possibilities for everyone. 

Learning something new every day keeps your mind sharp and engaged with the world. 

The weather outside can change quickly so it is good to stay prepared for anything. 

Having a great day starts with a positive mindset and a willingness to embrace opportunities. 

If you need help with something just ask and I will do my best to provide assistance. 

Time flies when you are having fun doing activities that you truly enjoy and love. 

Let us explore interesting topics together and discover new things along the way. 

Hello again my friend! It is always wonderful to see you returning for another chat. 

Are you ready to start an exciting conversation about whatever is on your mind today? 

Please feel free to tell me more about what you are thinking or working on recently. 

That sounds like a really great idea and I would love to hear more details about it. 

What do you think about the current situation and how do you feel it might develop? 

Let us take a short break if you need one because rest is important for productivity. 

How was your day so far? I hope it has been productive and filled with good moments. 

I really appreciate your help and cooperation as we work through this conversation together. 

See you later and take care until we speak again sometime soon in the near future. 

Welcome back to our chat! It is nice to have you here again for more conversation. 

Do you have any questions that I can help answer for you right now or later? 

Let us solve any problems you might have because most problems have solvable solutions. 

Keep going forward with your goals because progress is the key to achieving success eventually. 

  

1 + 1 = 2. 

2 + 2 = 4. 

3 + 3 = 6. 

4 + 4 = 8. 

5 + 5 = 10. 

6 + 6 = 12. 

7 + 7 = 14. 

8 + 8 = 16. 

9 + 9 = 18. 

10 + 10 = 20. 

11 + 11 = 22. 

12 + 12 = 24. 

13 + 13 = 26. 

14 + 14 = 28. 

15 + 15 = 30. 

16 + 16 = 32. 

17 + 17 = 34. 

18 + 18 = 36. 

19 + 19 = 38. 

20 + 20 = 40. 

25 + 25 = 50. 

30 + 30 = 60. 

35 + 35 = 70. 

40 + 40 = 80. 

45 + 45 = 90. 

50 + 50 = 100. 

1 * 1 = 1. 

2 * 2 = 4. 

3 * 3 = 9. 

4 * 4 = 16. 

5 * 5 = 25. 

6 * 6 = 36. 

7 * 7 = 49. 

8 * 8 = 64. 

9 * 9 = 81. 

10 * 10 = 100. 

2 * 1 = 2. 

2 * 2 = 4. 

2 * 3 = 6. 

2 * 4 = 8. 

2 * 5 = 10. 

2 * 6 = 12. 

2 * 7 = 14. 

2 * 8 = 16. 

2 * 9 = 18. 

2 * 10 = 20. 

3 * 1 = 3. 

3 * 2 = 6. 

3 * 3 = 9. 

3 * 4 = 12. 

3 * 5 = 15. 

3 * 6 = 18. 

3 * 7 = 21. 

3 * 8 = 24. 

3 * 9 = 27. 

3 * 10 = 30. 

4 * 1 = 4. 

4 * 2 = 8. 

4 * 3 = 12. 

4 * 4 = 16. 

4 * 5 = 20. 

4 * 6 = 24. 

4 * 7 = 28. 

4 * 8 = 32. 

4 * 9 = 36. 

4 * 10 = 40. 

5 * 1 = 5. 

5 * 2 = 10. 

5 * 3 = 15. 

5 * 4 = 20. 

5 * 5 = 25. 

5 * 6 = 30. 

5 * 7 = 35. 

5 * 8 = 40. 

5 * 9 = 45. 

5 * 10 = 50. 

10 - 1 = 9. 

10 - 2 = 8. 

10 - 3 = 7. 

10 - 4 = 6. 

10 - 5 = 5. 

10 - 6 = 4. 

10 - 7 = 3. 

10 - 8 = 2. 

10 - 9 = 1. 

10 - 10 = 0. 

20 - 10 = 10. 

30 - 10 = 20. 

40 - 10 = 30. 

50 - 10 = 40. 

60 - 10 = 50. 

70 - 10 = 60. 

80 - 10 = 70. 

90 - 10 = 80. 

100 - 10 = 90. 

100 - 20 = 80. 

100 - 30 = 70. 

100 - 40 = 60. 

100 - 50 = 50. 

100 - 60 = 40. 

100 - 70 = 30. 

100 - 80 = 20. 

100 - 90 = 10. 

100 - 100 = 0. 

50 + 50 = 100. 

25 + 25 = 50. 

10 + 90 = 100. 

  

Earth is the third planet from the Sun and the only astronomical object known to harbor life. This is made possible by Earth being an ocean world, the only one in the Solar System sustaining liquid surface water. Almost all of Earth's water is contained in its global ocean, covering 70.8% of Earth's crust. The remaining 29.2% of Earth's crust is land, most of which is located in the form of continental landmasses within Earth's land hemisphere. Most of Earth's land is at least somewhat humid and covered by vegetation, while large ice sheets at Earth's polar deserts retain more water than Earth's groundwater, lakes, rivers, and atmospheric water combined. Earth's crust consists of slowly moving tectonic plates, which interact to produce mountain ranges, volcanoes, and earthquakes. Earth has a liquid outer core that generates a magnetosphere capable of deflecting most of the destructive solar winds and cosmic radiation. 

Science is a systematic discipline that builds and organises knowledge in the form of testable hypotheses and predictions about the universe. Modern science is typically divided into two – or three – major branches: the natural sciences, which study the physical world, and the social sciences, which study individuals and societies. While referred to as the formal sciences, the study of logic, mathematics, and theoretical computer science are typically regarded as separate because they rely on deductive reasoning instead of the scientific method as their main methodology. Meanwhile, applied sciences are disciplines that use scientific knowledge for practical purposes, such as engineering and medicine. 

Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals. 

Python may refer to:. 

A computer is a machine that can be programmed to automatically carry out sequences of arithmetic or logical operations (computation). Modern digital electronic computers can perform generic sets of operations known as programs, which enable computers to perform a wide range of tasks. The term computer system may refer to a nominally complete computer that includes the hardware, operating system, software, and peripheral equipment needed and used for full operation, or to a group of computers that are linked and function together, such as a computer network or computer cluster. 

A school is an educational institution designed to provide learning environments for the teaching of students, usually under the direction of teachers. Most countries have systems of formal education, which is sometimes compulsory. In these systems, students progress through a series of schools that can be built and operated by both government and private organizations. The names for these schools vary by country but generally include primary school for young children and secondary school for teenagers who have completed primary education. An institution where higher education is taught is commonly called a university college or university. 

Cozmo is a miniature robot created by the defunct company Anki. Cozmo's base model, is a small, white and gray robot with red highlights. It makes use of distinct expressions, dubbed the "emotion engine", in order to mimic human emotion. Later editions came in red and white, gray and black and another in blue. 

Grok is a neologism coined by the American writer Robert A. Heinlein in his 1961 science fiction novel Stranger in a Strange Land. While the Oxford English Dictionary summarizes the meaning of grok as "to understand intuitively or by empathy, to establish rapport with", and "to empathize or communicate sympathetically (with); also, to experience enjoyment", Heinlein's concept of a human who comes to Earth in early adulthood after being born on the planet Mars is far more nuanced. 

Gemini most often refers to:Gemini (constellation), one of the constellations of the zodiac 

Gemini (astrology), an astrological sign. 

Physics is the scientific study of matter, its fundamental constituents, its motion and behavior through space and time, and the related entities of energy and force. It is one of the most fundamental scientific disciplines. A scientist who specializes in the field of physics is called a physicist. 

Chemistry is the scientific study of the properties and behavior of matter. It is a physical science within the natural sciences that studies the chemical elements that make up matter and compounds made of atoms, molecules and ions: their composition, structure, properties, behavior and the changes they undergo during reactions with other substances. Chemistry also addresses the nature of chemical bonds in chemical compounds. 

Biology is the scientific study of life and living organisms. It is a broad natural science that encompasses a wide range of fields and unifying principles that explain the structure, function, growth, origin, evolution, and distribution of life. Central to biology are five fundamental themes: the cell as the basic unit of life, genes and heredity as the basis of inheritance, evolution as the driver of biological diversity, energy transformation for sustaining life processes, and the maintenance of internal stability (homeostasis). 

Astronomy is a natural science that studies celestial objects and the phenomena that occur in the cosmos. It uses mathematics, physics, and chemistry to explain their origin and their overall evolution. Objects of interest include planets, moons, stars, nebulae, galaxies, meteoroids, asteroids, and comets. Relevant phenomena include supernova explosions, gamma ray bursts, quasars, blazars, pulsars, and cosmic microwave background radiation. More generally, astronomy studies everything that originates beyond Earth's atmosphere. Cosmology is the branch of astronomy that studies the universe as a whole. 

Geology is a branch of natural science concerned with the Earth and other astronomical bodies, the rocks of which they are composed, and the processes by which they change over time. The name comes from Ancient Greek  γῆ (gê) 'earth' and  λoγία (-logía) 'study of, discourse'. Modern geology significantly overlaps all other Earth sciences, including hydrology. It is integrated with Earth system science and planetary science. 

Ecology is the natural science of the relationships among living organisms and their environment. Ecology considers organisms at the individual, population, community, ecosystem, and biosphere levels. Ecology overlaps with the closely related sciences of biogeography, evolutionary biology, genetics, ethology, and natural history. 

Mathematics is a field of study that discovers and organizes methods, theories, and theorems that are developed and proved for the needs of empirical sciences and mathematics itself. There are many areas of mathematics, which include number theory, algebra, geometry, analysis, and set theory. 

Statistics is the discipline that concerns the collection, organization, analysis, interpretation, and presentation of data. In applying statistics to a scientific, industrial, or social problem, it is conventional to begin with a statistical population or a statistical model to be studied. Populations can be diverse groups of people or objects such as "all people living in a country" or "every atom composing a crystal". Statistics deals with every aspect of data, including the planning of data collection in terms of the design of surveys and experiments. 

Logic is the study of correct reasoning. It includes both formal and informal logic. Formal logic is the study of deductively valid inferences or logical truths. It examines how conclusions follow from premises based on the structure of arguments alone, independent of their topic and content. Informal logic is associated with informal fallacies, critical thinking, and argumentation theory. Informal logic examines arguments expressed in natural language whereas formal logic uses formal language. When used as a countable noun, the term "a logic" refers to a specific logical formal system that articulates a proof system. Logic plays a central role in many fields, such as philosophy, mathematics, computer science, and linguistics. 

Philosophy is a systematic study of general and fundamental questions concerning topics like existence, knowledge, mind, reason, language, and value. It is a rational and critical inquiry that reflects on its methods and assumptions. 

Psychology is the scientific study of the mind and behavior. Its subject matter includes the behavior of humans and nonhumans, both conscious and unconscious phenomena, and mental processes such as thoughts, feelings, and motives. Psychology is an academic discipline of immense scope, crossing the boundaries between the natural and social sciences. Biological psychologists seek an understanding of the emergent properties of brains, linking the discipline to neuroscience. As social scientists, psychologists aim to understand the behavior of individuals and groups. 

Sociology is the scientific study of human society that focuses on society, human social behavior, patterns of social relationships, social interaction, and aspects of culture associated with everyday life. The term sociology was coined in the late 18th century to describe the scientific study of society. Regarded as a part of both the social sciences and humanities, sociology uses various methods of empirical investigation and critical analysis to develop a body of knowledge about social order and social change. Sociological subject matter ranges from micro-level analyses of individual interaction and agency to macro-level analyses of social systems and social structure. Applied sociological research may be applied directly to social policy and welfare, whereas theoretical approaches may focus on the understanding of social processes and phenomenological method. 

Economics is a social science that studies the production, distribution, and consumption of goods and services. 

Political science is the social scientific study of politics. It deals with systems of governance and power, and the analysis of political activities, political thought, political behavior, and associated constitutions and laws. Specialists in the field are political scientists. Unlike political philosophy, which is primarily normative and concerns the theoretical and conceptual foundations of politics, political science emphasizes descriptive and explanatory of what is and favors empirical evidence over ethical judgements. 

History is the systematic study of the past, focusing primarily on the human past. As an academic discipline, it analyses and interprets evidence to construct narratives about what happened and explain why it happened. Some theorists categorize history as a social science, while others see it as part of the humanities or consider it a hybrid discipline. Similar debates surround the purpose of history—for example, whether its main aim is theoretical, to uncover the truth, or practical, to learn lessons from the past. In a more general sense, the term history refers not to an academic field but to the past itself, times in the past, or to individual texts about the past. 

Archaeology or archeology is the study of human activity through the recovery and analysis of material culture. The archaeological record consists of artifacts, architecture, biofacts or ecofacts, sites, and cultural landscapes. Archaeology can be considered both a social science and a branch of the humanities. It is usually considered an independent academic discipline, but may also be classified as part of anthropology, history or geography. The discipline involves surveying, excavation, and eventually analysis of data collected, to learn more about the past. In broad scope, archaeology relies on cross-disciplinary research. 

Anthropology is the scientific study of humanity that crosses biology and sociology, concerned with human behavior, human biology, cultures, societies, and linguistics, in both the present and past, including archaic humans. Social anthropology studies patterns of behaviour, while cultural anthropology studies cultural meaning, including norms and values. The term sociocultural anthropology is commonly used today. Linguistic anthropology studies how language influences social life. Biological anthropology studies the biology and evolution of humans and their close primate relatives. 

Linguistics is the scientific study of language. The areas of linguistic analysis are syntax, semantics (meaning), morphology, phonetics, phonology, and pragmatics. Subdisciplines such as biolinguistics and psycholinguistics bridge many of these divisions. 

Literature is any collection of written work. The term is also used more narrowly for writings considered an art form, especially novels, plays, and poems. It includes both print and digital writing. In recent centuries, the definition has expanded to include oral literature, much of which has been transcribed. Literature is a method of recording, preserving, and transmitting knowledge and entertainment. It can also have a social, psychological, spiritual, or political role. 

Art is a diverse range of cultural activity centered around works utilizing creative or imaginative talents, which are expected to evoke a worthwhile experience, generally through an expression of emotional power, conceptual ideas, technical proficiency, or beauty. 

Music theory is the study of theoretical frameworks for understanding the practices and possibilities of music. The Oxford Companion to Music describes three interrelated uses of the term "music theory": The first refers to the "rudiments" needed to understand music notation such as key signatures, time signatures, and rhythmic notation; the second is a study of scholars' views on music from antiquity to the present; the third is a sub-topic of musicology that "seeks to define processes and general principles in music". The musicological approach to theory differs from musical analysis "in that it takes as its starting-point not the individual work or performance but the fundamental materials from which it is built.". 

Engineering is the practice of using natural science, mathematics, and the engineering design process to solve problems within technology, increase efficiency and productivity, and improve systems. The traditional disciplines of engineering are civil, mechanical, electrical, and chemical. The academic discipline of engineering encompasses a broad range of more specialized subfields, and each can have a more specific emphasis for applications of mathematics and science. In turn, modern engineering practice spans multiple fields of engineering, which include designing and improving infrastructure, machinery, vehicles, electronics, materials, and energy systems. For related terms, see glossary of engineering. 

Electrical engineering is an engineering discipline concerned with the study, design, and application of equipment, devices, and systems that use electricity, electronics, and electromagnetism. It emerged as an identifiable occupation in the latter half of the 19th century after the commercialization of the electric telegraph, the telephone, and electrical power generation, distribution, and use. 

The American Society of Mechanical Engineers (ASME) is an American professional association that, in its own words, "promotes the art, science, and practice of multidisciplinary engineering and allied sciences around the globe" via "continuing education, training and professional development, codes and standards, research, conferences and publications, government relations, and other forms of outreach." ASME is thus an engineering society, a standards organization, a research and development organization, an advocacy organization, a provider of training and education, and a nonprofit organization. Founded as an engineering society focused on mechanical engineering in North America, ASME is today multidisciplinary and global. 

Civil Engineering is a professional engineering discipline that deals with the design, construction, and maintenance of the physical and naturally built environment, including public works such as roads, bridges, canals, dams, airports, sewage systems, pipelines, structural components of buildings, and railways. 

Computer science is the study of computation, information, and automation. Included broadly in the sciences, computer science spans theoretical disciplines to applied disciplines. An expert in the field is known as a computer scientist. 

Computer security is a subdiscipline within the field of information security. It focuses on protecting computer software, systems, and networks from threats that can lead to unauthorized information disclosure, theft or damage to hardware, software, or data, as well as to the disruption or misdirection of the services they provide. 

Data science is an interdisciplinary academic field that uses statistics, scientific computing, scientific methods, processing, scientific visualization, algorithms, and systems to extract or extrapolate knowledge from potentially noisy, structured, or unstructured data. 

Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Within a subdiscipline in machine learning, advances in the field of deep learning have allowed neural networks, a class of statistical algorithms, to surpass many previous machine learning approaches in performance. 

A neural network is a group of interconnected units called neurons that send signals to one another. Neurons can be either biological cells or mathematical models. While individual neurons are simple, many of them together in a network can perform complex tasks. There are two main types of neural networks.In neuroscience, a biological neural network is a physical structure found in brains and complex nervous systems – a population of nerve cells connected by synapses. 

In machine learning, an artificial neural network is a mathematical model used to approximate nonlinear functions. Artificial neural networks are used to solve artificial intelligence problems. 

A quantum computer is a computer that exploits superposed and entangled states. Quantum computers can be viewed as sampling from quantum systems that evolve in ways that may be described as operating on an enormous number of possibilities simultaneously, though still subject to strict computational constraints. By contrast, ordinary ("classical") computers operate according to deterministic rules. It is widely believed that a quantum computer could perform some calculations exponentially faster than any classical computer. For example, a large-scale quantum computer could break some widely used public-key cryptographic schemes and aid physicists in performing physical simulations. However, current hardware implementations of quantum computation are largely experimental and only suitable for specialized tasks. 

A blockchain is a distributed ledger with growing lists of records (blocks) that are securely linked together via cryptographic hashes. Each block contains a cryptographic hash of the previous block, a timestamp, and transaction data. Since each block contains information about the previous block, they effectively form a chain, with each additional block linking to the ones before it. Consequently, blockchain transactions are resistant to alteration because, once recorded, the data in any given block cannot be changed retroactively without altering all subsequent blocks and obtaining network consensus to accept these changes. 

Cryptography, or cryptology, is the practice and study of techniques for secure communication in the presence of adversarial behavior. More generally, cryptography is about constructing and analyzing protocols that prevent third parties or the public from reading private messages. Modern cryptography exists at the intersection of the disciplines of mathematics, computer science, information security, electrical engineering, digital signal processing, physics, and others. Core concepts related to information security are also central to cryptography. Practical applications of cryptography include electronic commerce, chip-based payment cards, digital currencies, computer passwords and military communications. 

Network, networking and networked may refer to:. 

An operating system (OS) is system software that manages computer hardware and software resources, and provides common services for computer programs. 

Cloud computing is defined by the ISO as "a paradigm for enabling network access to a scalable and elastic pool of shareable physical or virtual resources with self-service provisioning and administration on demand". It is commonly referred to as "the cloud". 

In computing, a database is an organized collection of data or a type of data store based on the use of a database management system (DBMS), the software that interacts with end users, applications, and the database itself to capture and analyze the data. The DBMS additionally encompasses the core facilities provided to administer the database. The sum total of the database, the DBMS and the associated applications can be referred to as a database system. Often the term "database" is also used loosely to refer to any of the DBMS, the database system or an application associated with the database. 

Genetics is the study of genes, genetic variation, and heredity in organisms. It is an important branch in biology because heredity is vital to organisms' evolution. Gregor Mendel, a Moravian Augustinian friar working in the 19th century in Brno, was the first to study genetics scientifically. Mendel studied "trait inheritance", patterns in the way traits are handed down from parents to offspring over time. He observed that organisms inherit traits by way of discrete "units of inheritance". This term, still used today, is a somewhat ambiguous definition of what is referred to as a gene. 

Neuroscience is the scientific study of the nervous system, its functions, and its disorders. It is a multidisciplinary science that combines physiology, anatomy, molecular biology, developmental biology, cytology, psychology, physics, computer science, chemistry, medicine, statistics, and mathematical modeling to understand the fundamental and emergent properties of neurons, glia, and neural circuits. The understanding of the biological basis of learning, memory, behavior, perception, and consciousness has been described by Eric Kandel as the "epic challenge" of the biological sciences. 

Medicine is the science and practice of caring for patients, managing the diagnosis, prognosis, prevention, treatment and palliation of their injury or disease, while promoting their health. Medicine encompasses a variety of health care practices which evolved to maintain and restore health through the prevention and treatment of illness. Contemporary medicine applies biomedical sciences, biomedical research, genetics, and medical technology to diagnose, treat, and prevent injury and disease, typically through various pharmaceuticals or surgery, but also through therapies such as psychotherapy, external splints and traction, medical devices, biologics, and ionizing radiation, amongst others. 

Public Health may refer to:Public health, promoting health through organized efforts and informed choices of society and individuals 

Public Health (journal), published by Elsevier for the Royal Society for Public Health 

Public Health a 2021 proposed comedy television series by Rob Tepper 

Public Health, a May 22, 2014 episode of Debatten, a Norwegian television series 

Public Health, a July 6, 2000 episode of Today's Environment, television series by Five Star Productions. 

Law is a set of rules that are created and are enforceable by governmental or societal institutions to regulate behavior, with its precise definition a matter of longstanding debate. It has been variously described as a science and as the art of justice. State-enforced laws can be made by a legislature, resulting in statutes; by the executive through decrees and regulations; or by judges' decisions, which form precedent in common law jurisdictions. An autocrat may exercise those functions within their realm. The creation of laws themselves may be influenced by a constitution, written or tacit, and the rights encoded therein. The law shapes politics, economics, history and society in various ways and also serves as a mediator of relations between people. 

Ethics is the philosophical study of moral phenomena. Also called moral philosophy, it investigates normative questions about what people ought to do or which behavior is morally right. Its main branches include normative ethics, applied ethics, and metaethics. 

Business is the practice of making one's living or making money by producing or buying and selling products. It is also "any activity or enterprise entered into for profit.". 

Finance refers to monetary resources and to the study and discipline of money, currency, assets and liabilities. As a subject of study, it is a field of business administration which involves the planning, organizing, leading, and controlling of an organization's resources to achieve its goals. Based on the scope of financial activities in financial systems, the discipline can be divided into personal, corporate, and public finance. 

Marketing is the act of acquiring, satisfying and retaining customers. It is one of the primary components of business management and commerce. 

Entrepreneurship is the creation or extraction of economic value by identifying and commercializing opportunities to deliver products or services, a process that typically requires considerable initiative and bears risk. This process may also encompass the pursuit of values that extend beyond mere economic considerations. 

Geopolitics is the study of the effects of Earth's geography on politics and international relations. Geopolitics usually refers to countries and relations between them. According to multiple researchers, the term is currently being used to describe a broad spectrum of concepts, in a general sense used as "a synonym for international political relations", but more specifically "to imply the global structure of such relations"; this usage builds on an "early-twentieth-century term for a pseudoscience of political geography" and other pseudoscientific theories of historical and geographic determinism. 

Climatology or climate science is the scientific study of Earth's climate, typically defined as weather conditions averaged over a period of at least 30 years. Climate concerns the atmospheric condition during an extended to indefinite period of time; weather is the condition of the atmosphere during a relative brief period of time. The main topics of research are the study of climate variability, mechanisms of climate changes and modern climate change. This topic of study is regarded as part of the atmospheric sciences and a subdivision of physical geography, which is one of the Earth sciences. Climatology includes some aspects of oceanography and biogeochemistry. 

Failed to fetch Energy Systems. 

Environmental science is an academic field that integrates the physical, biological, and mathematical sciences to study the environment and solve environmental problems. It uses an integrated, quantitative, and interdisciplinary approach to analyze environmental systems and emerged from the fields of natural history and medicine during the Enlightenment. It is considered interdisciplinary because it is an integration of various fields such as: biology, chemistry, physics, geology, engineering, sociology, and ecology. 

Astronautics is the practice of sending spacecraft beyond Earth's atmosphere into outer space. Spaceflight is one of its main applications and space science is its overarching field. 

Robotics is the interdisciplinary study and practice of the design, construction, operation, and use of robots. A roboticist is someone who specializes in robotics. 

Automation describes a wide range of technologies that reduce human intervention in processes, mainly by predetermining decision criteria, subprocess relationships, and related actions, as well as embodying those predeterminations in machines. Automation has been achieved by various means including mechanical, hydraulic, pneumatic, electrical, electronic devices, and computers, usually in combination. Complicated systems, such as modern factories, airplanes, and ships typically use combinations of all of these techniques. The benefits of automation includes labor savings, reducing waste, savings in electricity costs, savings in material costs, and improvements to quality, accuracy, and precision. 

Biotechnology is a multidisciplinary field that involves the integration of natural sciences and engineering sciences in order to achieve the application of organisms and parts thereof for products and services. Specialists in the field are known as biotechnologists. 

Nanotechnology is the manipulation of matter with at least one dimension sized from 1 to 100 nanometers (nm). At this scale, commonly known as the nanoscale, surface area and quantum mechanical effects become important in describing properties of matter. This definition of nanotechnology includes all types of research and technologies that deal with these special properties. It is common to see the plural form "nanotechnologies" as well as "nanoscale technologies" to refer to research and applications whose common trait is scale. An earlier understanding of nanotechnology referred to the particular technological goal of precisely manipulating atoms and molecules for fabricating macroscale products, now referred to as molecular nanotechnology. 

Materials science is an interdisciplinary field of researching and discovering materials. Materials engineering is an engineering field of finding uses for materials in other fields and industries. 

Cognitive science is the interdisciplinary, scientific study of the mind and its processes. It examines the nature, the tasks, and the functions of cognition. Mental faculties of concern to cognitive scientists include perception, memory, attention, reasoning, language, and emotion. To understand these faculties, cognitive scientists borrow from fields such as psychology, philosophy, artificial intelligence, neuroscience, linguistics, and anthropology. The typical analysis of cognitive science spans many levels of organization, from learning and decision-making to logic and planning; from neural circuitry to modular brain organization. One of the fundamental concepts of cognitive science is that "thinking can best be understood in terms of representational structures in the mind and computational procedures that operate on those structures.". 

Game theory is the study of mathematical models of strategic interactions. It has applications in many fields of social science, and is used extensively in economics, logic, systems science and computer science. Initially, game theory addressed two-person zero-sum games, in which a participant's gains or losses are exactly balanced by the losses and gains of the other participant. In the 1950s, it was extended to the study of non zero-sum games, and was eventually applied to a wide range of behavioral relations. It is now an umbrella term for the science of rational decision making in humans, animals, and computers. 

Information theory is the mathematical study of the quantification, storage, and communication of a particular type of mathematically defined information. The field was established and formalized by Claude Shannon in the 1940s, though early contributions were made in the 1920s through the works of Harry Nyquist and Ralph Hartley. It is at the intersection of electronic engineering, mathematics, statistics, computer science, neurobiology, physics, and electrical engineering. 

Employment is a relationship between two parties regulating the provision of paid labour services. Usually based on a contract, one party, the employer, which might be a corporation, a not-for-profit organization, a co-operative, or any other entity, pays the other, the employee, in return for carrying out assigned work. Employees work in return for wages, which can be paid on the basis of an hourly rate, by piecework or an annual salary, depending on the type of work an employee does, the prevailing conditions of the sector and the bargaining power between the parties. Employees in some sectors may receive gratuities, bonus payments or stock options. In some types of employment, employees may receive benefits in addition to payment. Benefits may include health insurance, housing, and disability insurance. 

Education is the transmission of knowledge and skills and the development of character traits. Formal education happens in a complex institutional framework, like public schools. Non-formal education is also structured but takes place outside the formal schooling system, while informal education is unstructured learning through daily experiences. Formal and non-formal education are divided into levels that include early childhood education, primary education, secondary education, and tertiary education. Other classifications focus on the teaching method, like teacher-centered and student-centered education, and on the subject, like science education, language education, and physical education. The term "education" can also refer to the mental states and qualities of educated people and the academic field studying educational phenomena. 

Sleep is a state of reduced mental and physical activity in which consciousness is altered and certain sensory activity is inhibited. During sleep, there is a marked decrease in muscle activity and interactions with the surrounding environment. While sleep differs from wakefulness in terms of the ability to react to stimuli, it still involves active brain patterns, making it more reactive than a coma or disorders of consciousness. 

A hobby is considered to be a regular activity that is done for enjoyment, typically during one's leisure time. Hobbies include collecting themed items and objects, engaging in creative and artistic pursuits, playing sports, or pursuing other amusements or avocations. Participation in hobbies encourages acquiring substantial skills and knowledge in that area. A list of hobbies changes with renewed interests and developing fashions, making it diverse and lengthy. Hobbies tend to follow trends in society. For example, stamp collecting was popular during the nineteenth and twentieth centuries as postal systems were the main means of communication; as of 2024, video games became more popular following technological advances. The advancing production, technology, and labour movements of the nineteenth century provided workers with more leisure time to engage in hobbies. Because of this, the efforts of people investing in hobbies has increased with time. 

Shopping is an activity in which a customer browses the available goods or services presented by one or more retailers with the potential intent to purchase a suitable selection of them. A typology of shopper types has been developed by scholars which identifies one group of shoppers as recreational shoppers, that is, those who enjoy shopping and view it as a leisure activity. 

Health has a variety of definitions, which have been used for different purposes over time. In general, it refers to physical and emotional well-being, especially that associated with normal functioning of the human body, absent of disease, pain, or injury. 

Family is a group of people related either by consanguinity or affinity. It forms the basis for social order. Ideally, families offer predictability, structure, and safety as members mature and learn to participate in the community. Historically, most human societies use family as the primary purpose of attachment, nurturance, and socialization. 

Leisure has often been defined as a quality of experience or as free time. Free time is time spent away from business, work, job hunting, domestic chores, and education, as well as necessary activities such as eating and sleeping. Leisure as an experience usually emphasizes dimensions of perceived freedom and choice. It is done for "its own sake", for the quality of experience and involvement. Other classic definitions include Thorstein Veblen's (1899) of "nonproductive consumption of time." Free time is not easy to define due to the multiplicity of approaches used to determine its essence. Different disciplines have definitions reflecting their common issues: for example, sociology on social forces and contexts and psychology as mental and emotional states and conditions. From a research perspective, these approaches have an advantage of being quantifiable and comparable over time and place. 

A grocery store (AE), grocery shop or grocer's shop (BE) or simply grocery is a retail store that primarily retails a general range of food products, which may be fresh or packaged. In everyday US usage, however, "grocery store" is a synonym for supermarket, and is not used to refer to other types of stores that sell groceries. In the UK, shops that sell food are distinguished as grocers or grocery shops. 

Physical fitness is a state of health and well-being and, more specifically, the ability to perform aspects of sports, occupations, and daily activities. Physical fitness is generally achieved through proper nutrition, moderate-vigorous physical exercise, and sufficient rest along with a formal recovery plan. 

Cleaning is the process of removing unwanted substances, such as dirt, dust, and other impurities, from an object or environment. Cleaning is often performed for aesthetic, hygienic, functional, safety, or environmental protection purposes. Cleaning occurs in many different contexts, and uses many different methods. Several occupations are devoted to cleaning. 

Laundry is the washing of clothing and other textiles, and, more broadly, their drying and ironing as well. Laundry has been part of history since humans began to wear clothes, so the methods by which different cultures have dealt with this universal human need are of interest to several branches of scholarship. 

Personal finance is the financial management that an individual or a family unit performs to budget, save, and spend monetary resources in a controlled manner, taking into account various financial risks and future life events. 

Telecommunication, often used in its plural form or abbreviated as telecom, is the transmission of information over a distance using electrical or electronic means, typically through cables, radio waves, or other communication technologies. These means of transmission may be divided into communication channels for multiplexing, allowing for a single medium to transmit several concurrent communication sessions. Long-distance technologies invented during the 19th, 20th and 21st centuries generally use electric power, and include the electrical telegraph, telephone, television, and radio. 

Social media are new media technologies that facilitate the creation, sharing and aggregation of content amongst virtual communities and networks. Common features include:Online platforms enable users to create and share content and participate in social networking. 

User-generated content—such as text posts or comments, digital photos or videos, and data generated through online interactions. 

Service-specific profiles that are designed and maintained by the social media organization. 

Social media helps the development of online social networks by connecting a user's profile with those of other individuals or groups. 

Television (TV) is a telecommunication medium for transmitting moving images and sound. Additionally, the term can refer to a physical television set rather than the medium of transmission. Television is a mass medium for advertising, entertainment, news, and sports. The medium is capable of more than "radio broadcasting", which refers to an audio signal sent to radio receivers. 

A birthday is the anniversary of the birth of a person or the figurative birth of an institution. Birthdays of people are celebrated in numerous cultures, often with birthday gifts, birthday cards, a birthday party, or a rite of passage. 

A wedding is a ceremony in which two people are united in marriage. Wedding traditions and customs vary greatly between cultures, ethnicities, races, religions, denominations, countries, social classes, and sexual orientations. Most wedding ceremonies involve an exchange of marriage vows by a couple; a presentation of a gift ; and a public proclamation of marriage by an authority figure or celebrant. Special wedding garments are often worn, and the ceremony is sometimes followed by a wedding reception. Music, poetry, prayers, or readings from religious texts or literature are also commonly incorporated into the ceremony, as well as superstitious customs. 

A funeral is a ceremony connected with the final disposition of a corpse, such as a burial, entombment or cremation with the attendant observances. Funerary customs comprise the complex of beliefs and practices used by a culture to remember and respect the dead, from interment, to various monuments, prayers, and rituals undertaken in their honour. Customs vary between cultures and religious groups. Funerals have both normative and legal components. Common secular motivations for funerals include mourning the deceased, celebrating their life, and offering support and sympathy to the bereaved; additionally, funerals may have religious aspects that are intended to help the soul of the deceased reach the afterlife, resurrection or reincarnation. 

Religion is a range of social-cultural systems, including designated behaviors and practices, ethics, morals, beliefs, worldviews, texts, sanctified places, prophecies, or organizations, that generally relate humanity to supernatural, transcendental, and spiritual elements—although there is no scholarly consensus over what precisely constitutes a religion. It is an essentially contested concept. Different religions may or may not contain various elements ranging from the divine, sacredness, faith, and a supernatural being or beings. 

Politics is the set of activities that are associated with making decisions in groups, or other forms of power relations among individuals, such as the distribution of status or resources. 

The branch of social science that studies politics and government is referred to as political science. 

History is the systematic study of the past, focusing primarily on the human past. As an academic discipline, it analyses and interprets evidence to construct narratives about what happened and explain why it happened. Some theorists categorize history as a social science, while others see it as part of the humanities or consider it a hybrid discipline. Similar debates surround the purpose of history—for example, whether its main aim is theoretical, to uncover the truth, or practical, to learn lessons from the past. In a more general sense, the term history refers not to an academic field but to the past itself, times in the past, or to individual texts about the past. 

Geography is the study of the lands, features, inhabitants, and phenomena of Earth. Geography is an all-encompassing discipline that seeks an understanding of Earth and its human and natural complexities—not merely where objects are, but also how they have changed and come to be. While geography is specific to Earth, many concepts can be applied more broadly to other celestial bodies in the field of planetary science. Geography has been called "a bridge between natural science and social science disciplines.". 

Science is a systematic discipline that builds and organises knowledge in the form of testable hypotheses and predictions about the universe. Modern science is typically divided into two – or three – major branches: the natural sciences, which study the physical world, and the social sciences, which study individuals and societies. While referred to as the formal sciences, the study of logic, mathematics, and theoretical computer science are typically regarded as separate because they rely on deductive reasoning instead of the scientific method as their main methodology. Meanwhile, applied sciences are disciplines that use scientific knowledge for practical purposes, such as engineering and medicine. 

Technology is the application of conceptual knowledge to achieve practical goals, especially in a reproducible way. The word technology can also mean the products resulting from such efforts, including both tangible tools such as utensils or machines, and intangible ones such as software. Technology plays a critical role in science, engineering, and everyday life. 

Art is a diverse range of cultural activity centered around works utilizing creative or imaginative talents, which are expected to evoke a worthwhile experience, generally through an expression of emotional power, conceptual ideas, technical proficiency, or beauty. 

Literature is any collection of written work. The term is also used more narrowly for writings considered an art form, especially novels, plays, and poems. It includes both print and digital writing. In recent centuries, the definition has expanded to include oral literature, much of which has been transcribed. Literature is a method of recording, preserving, and transmitting knowledge and entertainment. It can also have a social, psychological, spiritual, or political role. 

Philosophy is a systematic study of general and fundamental questions concerning topics like existence, knowledge, mind, reason, language, and value. It is a rational and critical inquiry that reflects on its methods and assumptions. 

Psychology is the scientific study of the mind and behavior. Its subject matter includes the behavior of humans and nonhumans, both conscious and unconscious phenomena, and mental processes such as thoughts, feelings, and motives. Psychology is an academic discipline of immense scope, crossing the boundaries between the natural and social sciences. Biological psychologists seek an understanding of the emergent properties of brains, linking the discipline to neuroscience. As social scientists, psychologists aim to understand the behavior of individuals and groups. 

Sociology is the scientific study of human society that focuses on society, human social behavior, patterns of social relationships, social interaction, and aspects of culture associated with everyday life. The term sociology was coined in the late 18th century to describe the scientific study of society. Regarded as a part of both the social sciences and humanities, sociology uses various methods of empirical investigation and critical analysis to develop a body of knowledge about social order and social change. Sociological subject matter ranges from micro-level analyses of individual interaction and agency to macro-level analyses of social systems and social structure. Applied sociological research may be applied directly to social policy and welfare, whereas theoretical approaches may focus on the understanding of social processes and phenomenological method. 

Economics is a social science that studies the production, distribution, and consumption of goods and services. 

Law is a set of rules that are created and are enforceable by governmental or societal institutions to regulate behavior, with its precise definition a matter of longstanding debate. It has been variously described as a science and as the art of justice. State-enforced laws can be made by a legislature, resulting in statutes; by the executive through decrees and regulations; or by judges' decisions, which form precedent in common law jurisdictions. An autocrat may exercise those functions within their realm. The creation of laws themselves may be influenced by a constitution, written or tacit, and the rights encoded therein. The law shapes politics, economics, history and society in various ways and also serves as a mediator of relations between people. 

An apple is the round, edible fruit of an apple tree. Fruit trees of the orchard or domestic apple, the most widely grown in the genus, are cultivated worldwide. The tree originated in Central Asia, where its wild ancestor, Malus sieversii, is still found. Apples have been grown for thousands of years in Eurasia before they were introduced to North America by European colonists. Apples have cultural significance in many mythologies and religions. 

A banana is an elongated, edible fruit—botanically a berry—produced by several kinds of large treelike herbaceous flowering plants in the genus Musa. In some countries, cooking bananas are called plantains, distinguishing them from dessert bananas. The fruit is variable in size, color and firmness, but is usually elongated and curved, with soft flesh rich in starch covered with a peel, which may have a variety of colors when ripe. It grows upward in clusters near the top of the plant. Almost all modern edible seedless (parthenocarp) cultivated bananas come from two wild species – Musa acuminata and Musa balbisiana, or their hybrids. 

Orange most often refers to:Orange (fruit), the fruit of the tree species  Citrus × sinensis 

Orange blossom, its fragrant flower 

Orange juice 

Orange (colour), the color of an orange fruit, occurs between red and yellow in the visible light spectrum 

Some other citrus or citrus-like fruit, see list of plants known as orange 

Orange (word), both a noun and an adjective in the English language. 

The garden strawberry is a widely grown hybrid plant cultivated worldwide for its fruit. The genus Fragaria, the strawberries, is in the rose family, Rosaceae. The fruit is appreciated for its aroma, bright red colour, juicy texture, and sweetness. It is eaten either fresh or in prepared foods such as jam, ice cream, and chocolates. Artificial strawberry flavourings and aromas are widely used in commercial products. Botanically, the strawberry is not a berry, but an aggregate accessory fruit. Each apparent 'seed' on the outside of the strawberry is actually an achene, a botanical fruit with a seed inside it. 

Blueberries are a widely distributed and widespread group of perennial flowering plants with blue or purple berries. They are classified in the section Cyanococcus within the genus Vaccinium. Commercial blueberries—both wild (lowbush) and cultivated (highbush)—are all native to North America. The highbush varieties were introduced into Europe during the 1930s. 

The raspberry is the edible fruit of several plant species in the genus Rubus of the rose family, most of which are in the subgenus Idaeobatus. The name also applies to these plants themselves. Raspberries are perennial with woody stems. 

The blackberry is an edible fruit produced by many species in the genus Rubus in the family Rosaceae, hybrids among these species within the subgenus Rubus, and hybrids between the subgenera Rubus and Idaeobatus. The taxonomy of blackberries has historically been confused because of hybridization and apomixis so that species have often been grouped together and called species aggregates. 

The pineapple is a tropical plant with an edible fruit; it is the most economically significant plant in the family Bromeliaceae. 

A mango is an edible stone fruit produced by the tropical tree Mangifera indica. It originated in the northeastern part of the Indian subcontinent, in what is now Bangladesh, northeastern India and Myanmar. M. indica has been cultivated in South and Southeast Asia since ancient times, resulting in two modern mango cultivar lineages: the "Indian" and the "Southeast Asian" types. Other species in the genus Mangifera also produce edible fruits called "mangoes," most of which are found in the Malesian ecoregion. 

The papaya, papaw, or pawpaw is the plant species Carica papaya, one of the 21 accepted species in the genus Carica of the family Caricaceae. Papaya is also the name of its fruit. It was first domesticated in Mesoamerica, within modern-day southern Mexico and Central America. It is grown in several countries in regions with a tropical climate. In 2024, India was the leading producer, accounting for 36% of the world total. 

A grape is a fruit, botanically a berry, of the deciduous woody vines of the flowering plant genus Vitis. Grapes are a non-climacteric type of fruit, generally occurring in clusters. 

The watermelon is a species of flowering plant in the family Cucurbitaceae, that has a large, edible fruit. It is a scrambling and trailing vine-like plant, and is widely cultivated worldwide, with more than 1,000 varieties. 

The cantaloupe is a type of true melon with sweet, aromatic, and usually orange flesh. Originally, cantaloup referred to the true cantaloupe or European cantaloupe with non- to slightly netted and often ribbed rind. Today, it also refers to the muskmelon with strongly netted rind, which is called cantaloupe in North America, rockmelon in Australia and New Zealand, and spanspek in Southern Africa. Cantaloupes range in mass from 0.5 to 5 kilograms. 

Honeydew may refer to:Honeydew (melon), a cultivar group of melon 

Honeydew (secretion), a sugar-rich sticky substance secreted by various animals 

Honeydew moth, a moth of Southern and Middle America 

Honeydew, California, United States, a town 

Honeydew, West Virginia, United States, an unincorporated community 

Honeydew (color), a pale shade of the color spring green 

Bunsen Honeydew, a fictional character from The Muppets franchise 

Honeydew (album), a 2008 album by Shawn Mullins 

Honeydew (film), a 2020 American horror film written and directed by Devereux Milburn 

Honey Dew Donuts, a Massachusetts-based franchise selling donuts and other breakfast foods 

Fuller's Organic Honey Dew, a brand of pale ale brewed by Fuller's Brewery 

Simon "Honeydew" Lane, a member of internet gaming group The Yogscast 

"Honeydew" , a 2023 episode of The Bear TV series 

  

. 

Kiwi most commonly refers to:Kiwi (bird), a flightless bird native to New Zealand 

Kiwi (nickname), an informal name for New Zealanders 

Kiwifruit, an edible hairy fruit with many seeds 

Kiwi dollar or New Zealand dollar, a unit of currency. 

The peach is a deciduous tree that bears edible juicy fruits with various characteristics. Most are simply called peaches, while the glossy-skinned, non-fuzzy varieties are called nectarines. Though from the same species, they are regarded commercially as different fruits. 

A plum is a fruit of some species in Prunus subg. Prunus. Dried plums are usually called prunes. 

A cherry is the fruit of many plants of the genus Prunus, and is a fleshy drupe. 

An apricot is a fruit, or the tree that bears the fruit, of several species in the genus Prunus. Usually an apricot is from the species Prunus armeniaca, but the fruits of the other species in Prunus sect. Armeniaca are also called apricots. In 2023, world production of apricots was 3.7 million tonnes, led by Turkey with 20% of the total. 

The pomegranate is a fruit-bearing, deciduous shrub in the family Lythraceae, subfamily Punicoideae, that grows to between 1.5–5 metres (5–16 ft) tall. Rich in symbolic and mythological associations in many cultures, it originated from the Iranian plateau including Iran, the Caucasus, Turkmenistan, Afghanistan and Pakistan. Pomegranate was first domesticated by ancient Iranians in the Persian plateau and nearby regions about 5,000 years ago. It is extensively cultivated for its fruit. 

The tomato is a plant whose fruit is an edible berry that is eaten as a vegetable. The tomato is a member of the nightshade family that includes tobacco, potato, and chili peppers. It originated from western South America, and may have been domesticated there, in Mexico, or in Central America. The Spanish introduced tomatoes to Eurasia in the Columbian exchange in the 16th century. 

The cucumber is a widely-cultivated creeping vine plant in the family Cucurbitaceae that bears cylindrical to spherical fruits, which are used as culinary vegetables. Considered an annual plant, there are three main types of cucumber—slicing, pickling, and seedless—within which several cultivars have been created. The cucumber originates in Asia extending from India, Nepal, Bangladesh, China, and Northern Thailand, but now grows on most continents, and many different types of cucumber are grown commercially and traded on the global market. In North America, the term wild cucumber refers to plants in the genera Echinocystis and Marah, though the two are not closely related. 

The carrot is a root vegetable, typically orange in colour, though heirloom variants including purple, black, red, white, and yellow cultivars exist, all of which are domesticated forms of the wild carrot, Daucus carota, native to Europe and Southwestern Asia. The plant probably originated in Iran and was originally cultivated for its leaves and seeds. 

Broccoli is an edible green plant in the cabbage family whose large flowering head, stalk and small associated leaves are eaten as a vegetable. Broccoli is classified in the Italica cultivar group of the species Brassica oleracea. Broccoli has large flower heads, or florets, usually dark green, arranged in a tree-like structure branching out from a thick stalk, which is usually light green. Leaves surround the mass of flower heads. Broccoli resembles cauliflower, a different but closely related cultivar group of the same Brassica species. 

Cauliflower is one of several vegetables cultivated from the species Brassica oleracea in the genus Brassica, which is in the Brassicaceae family. Cauliflower usually grows with one main stem that carries a large, rounded "head" made of tightly clustered, immature white or off-white flower buds called the "curd". Typically, only the "head" is eaten. 

Spinach is a leafy green flowering plant native to Central and Western Asia. It is of the order Caryophyllales, family Amaranthaceae, subfamily Chenopodioideae. Its leaves are a common vegetable consumed either fresh, cooked or after storage. The taste differs considerably between cooked and raw: the high oxalate content may be reduced by steaming. 

Kale, also called leaf cabbage, belongs to a group of cabbage cultivars primarily grown for their edible leaves, but it is also used as an ornamental plant. Its multiple different cultivars vary quite a bit in appearance; the leaves can be bumpy, curly, or flat, and the color ranges from purple to green. 

Lettuce is an annual plant of the family Asteraceae mostly grown as a leaf vegetable. The leaves are most often used raw in green salads, although lettuce is also seen in other kinds of food, such as sandwiches, wraps and soups; it can also be grilled. Its stem and seeds are sometimes used; celtuce is one variety grown for its stems, which are eaten either raw or cooked. In addition to its main use as a leafy green, it has also gathered religious and medicinal significance over centuries of human consumption. Europe and North America originally dominated the market for lettuce, but by the late 20th century the consumption of lettuce had spread throughout the world. In 2023, world production of lettuce was 28 million tonnes, led by China with 53% of the total. 

Eruca sativa is an edible annual plant in the family Brassicaceae. Other common names include salad rocket, garden rocket, colewort, roquette, ruchetta, rucola, rucoli, and rugula. 

Zucchini, courgette, or Cucurbita pepo var. cylindrica is a summer squash, a vining herbaceous plant whose fruit are harvested when their immature seeds and epicarp (rind) are still soft and edible. It is closely related, but not identical, to the marrow; its fruit may be called marrow when mature. 

Eggplant, aubergine, brinjal, or baigan is a plant species in the nightshade family Solanaceae. Solanum melongena is grown worldwide for its edible fruit, typically used as a vegetable in cooking. 

The bell pepper is the fruit of plants in the Grossum Group of the species Capsicum annuum. Cultivars of the plant produce fruits in different colors, including red, yellow, orange, green, white, and purple. Bell peppers are sometimes grouped with less pungent chili varieties as "sweet peppers". While they are botanically fruits—classified as berries—they are commonly used as a vegetable ingredient or side dish. Other varieties of the genus Capsicum are categorized as chili peppers when they are cultivated for their pungency, including some varieties of Capsicum annuum. 

The onion, also known as the bulb onion or common onion, is a vegetable that is the most widely cultivated species of the genus Allium. The shallot is a botanical variety of the onion which was classified as a separate species until 2011. The onion's close relatives include garlic, scallion, leek, and chives. 

Garlic is a species of bulbous flowering plants in the genus Allium. Its close relatives include the onion, shallot, leek, chives, Welsh onion, and Chinese onion. Garlic is native to central and western Asia, stretching from the Black Sea through the southern Caucasus, northeastern Iran, and the Hindu Kush. It has naturalized in many other parts of the world, including Mediterranean Europe and China. There are two subspecies and hundreds of varieties of garlic. 

Ginger is a flowering plant whose rhizome, ginger root or ginger, is widely used as a spice and a folk medicine. It is an herbaceous perennial that grows annual pseudostems about one meter tall, bearing narrow leaf blades. The inflorescences bear flowers having pale yellow petals with purple edges, and arise directly from the rhizome on separate shoots. 

The potato is a starchy tuberous vegetable native to the Americas that is consumed as a staple food in many parts of the world. Potatoes are underground stem tubers of the plant Solanum tuberosum, a perennial in the nightshade family Solanaceae. 

The sweet potato or sweetpotato is a dicotyledonous plant in the morning glory family, Convolvulaceae. Its sizeable, starchy, sweet-tasting tuberous roots are used as a root vegetable, which is a staple food in parts of the world. Cultivars of the sweet potato have been bred to bear tubers with flesh and skin of various colors. Moreover, the young shoots and leaves are occasionally eaten as greens. The sweet potato and the potato are only distantly related, both being in the order Solanales. Although darker sweet potatoes are often known as yams in parts of North America, they are even more distant from actual yams, which are monocots in the order Dioscoreales. 

Maize, also known as corn in North American English, is a tall stout grass that produces cereal grain. The leafy stalk of the plant gives rise to male inflorescences or tassels which produce pollen, and female inflorescences called ears. The ears yield grain, known as kernels or seeds. In modern commercial varieties, these are usually yellow or white; other varieties can be of many colors. Maize was domesticated by indigenous peoples in southern Mexico about 9,000 years ago from wild teosinte. Native Americans planted it alongside beans and squashes in the Three Sisters polyculture. 

Could not find summary for "Green Beans". 

Asparagus or garden asparagus is a perennial flowering plant species in the genus Asparagus native to Eurasia. Widely cultivated as a vegetable crop, its young shoots are used as a spring vegetable. 

Celery is a cultivated plant belonging to the species Apium graveolens in the family Apiaceae that has been used as a vegetable since ancient times. 

A mushroom is the fleshy, spore-bearing fruiting body of a fungus, typically produced above ground on soil or another food source. A toadstool generally refers to a poisonous mushroom. 

The avocado, alligator pear or avocado pear is an evergreen tree in the laurel family (Lauraceae). It is native to the Americas, with archaeological evidence of early human avocado use dating back thousands of years across various regions of Central and South America. It was prized for its large and unusually oily fruit. The native range of avocado extends from Mexico to Peru, encompassing much of Central America and parts of northern and western South America. 

Lime most commonly refers to:Lime (fruit), a green citrus fruit 

Lime (material), inorganic materials containing calcium, usually calcium oxide or calcium hydroxide 

Lime (color), a color between yellow and green. 

The lemon is a species of small evergreen tree in the Citrus genus of the flowering plant family Rutaceae. A true lemon is a hybrid of the citron and the bitter orange. Its origins are uncertain, but some evidence suggests lemons originated during the 1st millennium BC in what is now northeastern India. Some other citrus fruits are called lemon. 

The grapefruit is a subtropical citrus tree known for its relatively large, sour to semi-sweet, somewhat bitter fruit. The flesh of the fruit is segmented and varies in color from pale yellow to dark red. 

Pears are fruits produced and consumed around the world, growing on a tree and are harvested in late summer into mid-autumn. The pear tree and shrub are a species of genus Pyrus, in the family Rosaceae, bearing the pomaceous fruit of the same name. Several species of pears are valued for their edible fruit and juices, while others are cultivated as trees. 

The coconut is a member of the palm family (Arecaceae) and the only living species of the genus Cocos. The term "coconut" can denote the whole coconut palm tree or the large hard fruit. Originally native to Central Indo-Pacific, they are ubiquitous in coastal tropical regions. 

Passiflora edulis, commonly known as passion fruit, is a vine species of passion flower. The fruit is a pepo, a type of botanical berry, round to oval, either yellow or dark purple at maturity, with a soft to firm, juicy interior filled with numerous seeds. 

Lychee is a monotypic taxon and the sole member in the genus Litchi in the soapberry family, Sapindaceae. 

The fruit is edible and has a sweet, mildly tart flavor and a distinctive floral aroma often described as rose-like. 

The durian is the edible fruit of several tree species belonging to the genus Durio. There are 30 recognised species, at least nine of which produce edible fruit. Durio zibethinus, native to Borneo, Sumatra, and the Malay Peninsula, is the only species available on the international market. It has over 300 named varieties in Thailand and over 200 in Malaysia as of 2021. Other species are sold in their local regions. 

Guava, also known as the 'guava-pear' in various regions, is a common tropical fruit cultivated in many tropical and subtropical regions. The common guava Psidium guajava is a small tree in the myrtle family (Myrtaceae), native to Mexico, Central America, the Caribbean and northern South America. 

Carambola, also known as star fruit, is the fruit of Averrhoa carambola, a species of tree native to tropical Southeast Asia. The edible fruit has distinctive ridges running down its sides. When cut in cross-section, it resembles a star, giving it the name of star fruit. The entire fruit is edible, usually raw, and may be cooked or made into relishes, preserves, garnish, and juices. It is commonly consumed in Southeast Asia, South Asia, the South Pacific, Micronesia, parts of East Asia, the United States, parts of Latin America, and the Caribbean. The tree is cultivated throughout tropical areas of the world. 

Pitaya, pitahaya or commonly known as dragon fruit is the fruit of several cactus species indigenous to the region of southern Mexico and along the Pacific coasts of Guatemala, Costa Rica, and El Salvador. Pitaya is cultivated in East Asia, South Asia, Southeast Asia, continental America, the Caribbean, Australia, Brazil, Madeira (Portugal), and throughout tropical and subtropical regions of the world. 

Rice is a cereal grain and in its domesticated form is the staple food of over half of the world's population, particularly in Asia and Africa. Rice is the seed of the grass species Oryza sativa —or, much less commonly, Oryza glaberrima. Asian rice was domesticated in China some 13,500 to 8,200 years ago; African rice was domesticated in Africa about 3,000 years ago. Rice has become commonplace in many cultures worldwide; in 2023, 800 million tons were produced, placing it third after sugarcane and maize. Only some 8% of rice is traded internationally. China, India, and Indonesia are the largest consumers of rice. A substantial amount of the rice produced in developing nations is lost after harvest through factors such as poor transport and storage. Rice yields can be reduced by pests including insects, rodents, and birds, as well as by weeds, and by diseases such as rice blast. Traditional rice polycultures such as rice-duck farming, and modern integrated pest management seek to control damage from pests in a sustainable way. 

Pasta is a type of food typically made from an unleavened dough of wheat flour mixed with water or eggs, and formed into sheets or other shapes, then cooked by boiling or baking. Pasta was originally only made with durum, although the definition has been expanded to include alternatives for a gluten-free diet, such as rice flour, or legumes such as beans or lentils. Pasta is believed to have developed independently in Italy and is a staple food of Italian cuisine, with evidence of Etruscans making pasta as early as 400 BCE in Italy. 

Bread is a baked food product made from water, flour, and often yeast. It is a staple food across the world, particularly in Europe and the Middle East. Throughout recorded history and around the world, it has been an important part of many cultures' diets. It is one of the oldest human-made foods, having been of significance since the dawn of agriculture, and plays an essential role in both religious rituals and secular culture. 

A tortilla is a thin, circular unleavened flatbread from Mesoamerica originally made from masa, and now also from wheat flour. 

The oat, sometimes called the common oat, is a species of cereal grass (Avena) grown for fodder and for its seed, which is known by the same name. Oats appear to have been domesticated as a secondary crop, as their seeds resembled those of other cereals closely enough for them to be included by early cultivators. Oats tolerate cold winters less well than cereals such as wheat, barley, and rye, but need less summer heat and more rain, making them important in areas such as Northwest Europe that have cool, wet summers. They can tolerate low-nutrient and acid soils. Oats grow thickly and vigorously, allowing them to outcompete many weeds, and compared to other cereals are relatively free from diseases. 

Quinoa is a flowering plant in the amaranth family. It is a herbaceous annual plant grown as a crop primarily for its edible seeds; the seeds are high in protein, dietary fiber, B vitamins and dietary minerals especially potassium and magnesium in amounts greater than in many grains. Quinoa is not a grass but rather a pseudocereal botanically related to spinach and amaranth, and originated in the Andean region of northwestern South America. It was first used to feed livestock 5,200–7,000 years ago, and for human consumption 3,000–4,000 years ago in the Lake Titicaca basin of Bolivia and Peru. 

Barley, a member of the grass family, is a major cereal grain grown in temperate climates globally. One of the first cultivated grains, it was domesticated in the Fertile Crescent around 9000 BC, giving it nonshattering spikelets and making it much easier to harvest. Its use then spread throughout Eurasia by 2000 BC. Barley prefers relatively low temperatures and well-drained soil to grow. It is relatively tolerant of drought and soil salinity, but is less winter-hardy than wheat or rye. 

The lentil is an annual legume grown for its lens-shaped edible seeds or pulses, also called lentils. It is about 40 cm (16 in) tall, and the seeds grow in pods, usually with two seeds in each. 

The chickpea or chick pea is an annual legume of the family Fabaceae, subfamily Faboideae, cultivated for its edible seeds. Its different types are variously known as gram, Bengal gram, chana dal, garbanzo, garbanzo bean, or Egyptian pea. It is one of the earliest cultivated legumes, the oldest archaeological evidence of which was found in Syria. 

Could not find summary for "Black Beans". 

The kidney bean is a variety of the common bean ; it has such a common name owing to its resemblance to a human kidney. 

Tofu  or bean curd is a food prepared by pressing the curds of coagulated soy milk into solid white blocks of varying softness: silken, soft, firm, and extra firm. 

Tempeh or tempe is a traditional Indonesian food made from fermented soybeans. It is made by a natural culturing and controlled fermentation process that binds soybeans into a cake form. A fungus, Rhizopus oligosporus or Rhizopus oryzae, is used in the fermentation process and is also known as tempeh starter. 

The chicken is a domesticated form of the red junglefowl, originally native to Southeast Asia. It was first domesticated around 8,000 years ago and is one of the most common and widespread domesticated animals in the world. Chickens are primarily kept for their meat and eggs, though they are also kept as pets. 

Beef is the culinary name for meat from cattle. Beef can be prepared in various ways; cuts are often used for steak, which can be cooked to varying degrees of doneness, while trimmings are often ground or minced, as found in most hamburgers. Beef contains protein, iron, and vitamin B12. Along with other kinds of red meat, high consumption is associated with an increased risk of colorectal cancer and cardiovascular disease, especially when processed. Beef has a high environmental impact, being a primary driver of deforestation with the highest greenhouse gas emissions of any agricultural product. 

Pork is the culinary name for the meat of the pig. It is the second most commonly consumed type of meat worldwide, following poultry, with evidence of pig husbandry dating back to 8000–9000 BCE. 

Turkey, officially the Republic of Türkiye, is a country mainly located in Anatolia in West Asia, with a smaller part called East Thrace in Southeast Europe. It borders the Black Sea to the north; Georgia, Armenia, Azerbaijan, and Iran to the east; Iraq, Syria, and the Mediterranean Sea to the south; and the Aegean Sea, Greece, and Bulgaria to the west. Turkey is home to over 86 million people; most are ethnic Turks, while Kurds are the largest ethnic minority. Officially a secular state, Turkey has a Muslim-majority population. Ankara is Turkey's capital and second-largest city. Istanbul is its largest city and economic center. Other major cities include İzmir, Bursa, and Antalya. 

Salmon are any of several commercially important species of euryhaline ray-finned fish from the genera Salmo and Oncorhynchus of the family Salmonidae, native to tributaries of the North Atlantic (Salmo) and North Pacific (Oncorhynchus) basins. Salmon is a colloquial or common name used for fish in this group, but is not a scientific name. Other closely related fish in the same family include trout, char, grayling, whitefish, lenok and taimen, all coldwater fish of the subarctic and cooler temperate regions with some sporadic endorheic populations in Central Asia. 

A tuna is a saltwater fish that belongs to the tribe Thunnini, a subgrouping of the Scombridae (mackerel) family. The Thunnini comprise 15 species across five genera, the sizes of which vary greatly, ranging from the bullet tuna up to the Atlantic bluefin tuna, which averages 2 m (6.6 ft) and is believed to live up to 50 years. 

A shrimp is a common name typically used for crustaceans with an elongated body and a primarily swimming mode of locomotion – usually decapods belonging to the Caridea or Dendrobranchiata, although some crustaceans outside of this order are also referred to as "shrimp". 

Crabs are decapod crustaceans, either the Brachyura or various groups within the closely related Anomura, characterised by having a heavily armoured shell, their tail segments concealed under the body, the ability to run sideways, and the habit of hiding in rocky crevices. They do not form a single natural group or clade, but have convergently evolved multiple times from the ancestral decapod body plan through carcinisation, the process of creating this set of characteristics. As a group, they are thus polyphyletic, meaning they have multiple evolutionary origins. 

Lobsters are malacostracan decapod crustaceans of the family Nephropidae or its synonym Homaridae. They have long bodies with muscular tails and live in crevices or burrows on the sea floor. Three of their five pairs of legs have claws, including the first pair, which are usually much larger than the others. Highly prized as seafood, lobsters are economically important and are often one of the most profitable commodities in the coastal areas they populate. 

An egg is an organic vessel in which an embryo begins to develop. 

Milk is a usually white liquid food produced by the mammary glands of lactating mammals. It is the primary source of nutrition for young mammals before they are able to digest solid food. Milk contains many nutrients, including calcium and protein, as well as lactose and saturated fat; the enzyme lactase is needed to break down lactose. Immune factors and immune-modulating components in milk contribute to milk immunity. The first milk, which is called colostrum, contains antibodies and immune-modulating components that strengthen the immune system against many diseases. 

Cheddar cheese is a natural cheese that is relatively hard, off-white, and sometimes sharp-tasting. It originates from the village of Cheddar in Somerset, South West England. 

Mozzarella is a semi-soft non-aged cheese prepared using the pasta filata ('stretched-curd') method. It originated in southern Italy. 

Yogurt is a food produced by bacterial fermentation of milk. Fermentation of sugars in the milk by these bacteria produces lactic acid, which acts on milk protein to give yogurt its texture and characteristic tart flavor. Cow's milk is most commonly used to make yogurt. Milk from water buffalo, goats, ewes, mares, camels, and yaks is also used to produce yogurt. The milk used may be homogenized or not. It may be pasteurized or raw. Each type of milk produces substantially different results. 

Butter is a dairy product made from the fat and protein components of churned cream. It is a semi-solid emulsion at room temperature, consisting of approximately 81% butterfat. It is used at room temperature as a spread, melted as a condiment, and used as a fat in baking, sauce-making, pan frying, and other cooking procedures. 

The almond is a species of tree from the genus Prunus. Along with the peach, it is classified in the subgenus Amygdalus, distinguished from the other subgenera by corrugations on the shell (endocarp) surrounding the seed. 

A walnut is the edible seed of any tree of the genus Juglans, particularly the Persian or English walnut, Juglans regia. They are accessory fruit because the outer covering of the fruit is technically an involucre and thus not morphologically part of the carpel; this means it cannot be a drupe but is instead a drupe-like nut. 

Cashew is the common name of a tropical evergreen tree Anacardium occidentale, in the family Anacardiaceae. It is the source of the cashew nut and the cashew apple. The tree can grow as tall as 14 meters. 

Peanuts is a syndicated daily and Sunday American comic strip written and illustrated by Charles M. Schulz. The strip originally ran from 1950 to 2000, continuing in reruns afterward. Peanuts is regarded as one of the most popular and influential comic strips in history, with 17,897 strips published in all, making it "arguably the longest story ever told by one human being". At the time of Schulz's death in 2000, Peanuts ran in over 2,600 newspapers, with a readership of roughly 355 million across 75 countries, and had been translated into 21 languages. It helped to cement the four-panel gag strip as the standard in the United States, and together with its merchandise earned Schulz more than $1 billion. Following successful animated television and stage-theatrical adaptations over the years, five animated theatrical films have been released. 

Sunflower seeds are the seeds of the sunflower (Helianthus). 

Could not find summary for "Pumpkin Seeds". 

Olive oil is a vegetable oil obtained by pressing whole olives and extracting the oil. 

Honey is a sweet and viscous substance made by several species of bees, the best-known of which are honey bees. Honey is made and stored to nourish bee colonies. Bees produce honey by gathering and then refining the sugary secretions of plants or the secretions of other insects, like the honeydew of aphids. This refinement takes place both within individual bees, through regurgitation and enzymatic activity, and during storage in the hive, through water evaporation that concentrates the honey's sugars until it is thick and viscous. 

Maple syrup is a sweet syrup made from the sap of maple trees. In cold climates these trees store starch in their trunks and roots before winter; the starch is then converted to sugar that rises in the sap in late winter and early spring. Maple trees are tapped by drilling holes into their trunks and collecting the sap, which is heated to evaporate much of the water, leaving the concentrated syrup. 

Chocolate is a food made from roasted and ground cocoa beans that can be a liquid, solid, or paste, either by itself or to flavor other foods. Cocoa beans are the processed seeds of the cacao tree. They are usually fermented to develop the flavor, then dried, cleaned, and roasted. The shell is removed to reveal nibs, which are ground to chocolate liquor The liquor can be processed to separate its two components, cocoa solids and cocoa butter, or shaped and sold as unsweetened baking chocolate. By adding sugar, sweetened chocolates are produced, which can be sold simply as dark chocolate, or, with the addition of milk, can be made into milk chocolate. Making milk chocolate with cocoa butter and without cocoa solids produces white chocolate. 

Vanilla is a spice derived from orchids of the genus Vanilla, primarily obtained from the seed pods of the flat-leaved New World vanilla (V. planifolia). 

Cinnamon is a spice obtained from the inner bark of several tree species from the genus Cinnamomum. Cinnamon is used mainly as an aromatic condiment and flavouring additive in a wide variety of cuisines, in particular sweet and savoury dishes such as biscuits, breakfast cereals, snack foods, bagels, teas, hot chocolate, and traditional foods. The aroma and flavour of cinnamon derive from its essential oil and principal component, cinnamaldehyde, as well as numerous other constituents, including eugenol. 

Basil, also called great basil, is a culinary herb of the family Lamiaceae (mints). It is a tender plant, and is used in cuisines worldwide. In Western cuisine, the generic term "basil" refers to the variety also known as Genovese basil or sweet basil. Basil is native to tropical regions from Central Africa to Southeast Asia. In temperate climates basil is treated as an annual plant, but it can be grown as a short-lived perennial or biennial in warmer horticultural zones with tropical or Mediterranean climates. 

Oregano is a species of flowering plant in the mint family, Lamiaceae. It was native to the Mediterranean region, but widely naturalised elsewhere in the temperate Northern Hemisphere. 

Parsley, or garden parsley, is a species of flowering plant in the family Apiaceae that is native to the Balkans. It has been introduced and naturalized in Europe and elsewhere in the world with suitable climates, and is widely cultivated as a herb and a vegetable. 

Mint or The Mint may refer to:. 

Salvia rosmarinus, synonym Rosmarinus officinalis, commonly known as rosemary, is a shrub with fragrant, evergreen, needle-like leaves and purple or sometimes white, pink, or blue flowers. It is a member of the mint family, Lamiaceae. 

Thyme is a culinary herb consisting of the dried aerial parts of some members of the genus Thymus of flowering plants in the mint family Lamiaceae. Thymes are native to Eurasia and north Africa. Thymes have culinary, medicinal, and ornamental uses. The species most commonly cultivated and used for culinary purposes is Thymus vulgaris, native to Southeast Europe. 

A telephone, commonly shortened to phone, is a telecommunications device that enables two or more users to conduct a conversation when they are too far apart to be easily heard directly. A telephone converts sound, typically and most efficiently the human voice, into electronic signals that are transmitted via cables and other communication channels to another telephone which reproduces the sound to the receiving user. The term is derived from Ancient Greek: τῆλε, romanized: tēle, lit. 'far' and φωνή, together meaning distant voice. 

A laptop is a portable personal computer (PC). Laptops typically have a clamshell form factor with a flat-panel screen on the inside of the upper lid and an alphanumeric keyboard and pointing device on the inside of the lower lid. Most of the computer's internal hardware is in the lower part, under the keyboard, although many modern laptops have a built-in webcam at the top of the screen, and some even feature a touchscreen display. In most cases, unlike tablet computers which run on mobile operating systems, laptops tend to run on desktop operating systems, which were originally developed for desktop computers. 

Tablet may refer to:. 

Keyboard may refer to:. 

A mouse is a small rodent. Characteristically, mice are known to have a pointed snout, small rounded ears, a body-length scaly tail, and a high breeding rate. The best known mouse species is the common house mouse. Mice are also popular as pets. In some places, certain kinds of field mice are locally common. They are known to invade homes for food and shelter. 

Monitor or monitor may refer to:. 

Headphones are a pair of small loudspeaker drivers worn on or around the head over a user's ears. They are electroacoustic transducers, which convert an electrical signal to a corresponding sound. Headphones let a single user listen to an audio source privately, in contrast to a loudspeaker, which emits sound into the open air for anyone nearby to hear. Headphones are also known as earphones or, colloquially, cans. Circumaural and supra-aural headphones use a band over the top of the head to hold the drivers in place. Another type, known as earbuds or earpieces, consists of individual units that plug into the user's ear canal; within that category have been developed cordless air buds using wireless technology. A third type are bone conduction headphones, which typically wrap around the back of the head and rest in front of the ear canal, leaving the ear canal open. In the context of telecommunication, a headset is a combination of a headphone and microphone. 

Charger or Chargers may refer to:. 

A backpack, also called knapsack, schoolbag, rucksack, pack, booksack, bookbag, haversack, packsack, or backsack, is in its simplest frameless form, a fabric sack carried on one’s back and secured with two straps that go over the shoulders, and is used to carry goods from one place to another. It can feature an external or internal frame to transfer heavy loads off the user’s shoulders and onto their hips, reducing strain and increasing comfort on long hikes with heavy gear. 

A wallet is a flat case or pouch, often used to carry small personal items such as physical currency, debit cards, and credit cards; identification documents such as driving licence, identification card, club card; photographs, transit pass, business cards and other paper or laminated cards. Wallets are generally made of fabric or leather, and they are usually pocket-sized and foldable. 

Key, Keys, The Key or The Keys may refer to:. 

A pen is a common writing instrument that applies ink to a surface, typically paper, for writing or drawing. Early pens such as reed pens, quill pens, dip pens and ruling pens held a small amount of ink on a nib or in a small void or cavity that had to be periodically recharged by dipping the tip of the pen into an inkwell. 

Today, such pens find only a small number of specialized uses, such as in illustration and calligraphy. Reed pens, quill pens and dip pens, which were used for writing, have been replaced by ballpoint pens, rollerball pens, fountain pens and felt or ceramic tip pens. 

A pencil is a writing or drawing implement with a solid pigment core in a protective casing that reduces the risk of core breakage and keeps it from marking the user's hand. 

A notebook is a book or stack of paper pages that are often ruled and used for purposes such as note-taking, journaling, or other writing, drawing, or scrapbooking and more. 

  

  

  

Paper is a thin sheet of matted cellulose fibers. Largely derived from lignocellulose, paper is created from a pulp dissolved into a slurry that is drained and dried into sheets. Different types of paper are defined by constituent fiber, paper pulp, sizing, coating, paper size, paper density and grammage. 

An eraser is an article of stationery that is used for removing marks from paper or skin. Erasers have a rubbery consistency and come in a variety of shapes, sizes, and colors. Some pencils have an eraser on one end. Less expensive erasers are made from synthetic rubber and synthetic soy-based gum, but more expensive or specialized erasers are made from vinyl, plastic, or gum-like materials. 

A highlighter, also called a fluorescent pen, is a type of writing device used to bring attention to sections of text by marking them with a vivid, translucent colour. 

A typical highlighter is fluorescent yellow, with the colour coming from pyranine. Different compounds, such as rhodamines are used for other colours. 

A ruler is an instrument used to make length measurements, whereby a length is read from a series of markings called "rules" along an edge of the device. Alternatively, it is called a rule, scale, line gauge, or metre/meter stick. Usually, the instrument is rigid and the edge itself is a straightedge, which additionally allows one to draw straighter lines. Rulers are an important tool in geometry, geography and mathematics. They have been used since at least 2650 BC. 

Scissors or shears are hand-operated cutting tools that consists of a pair of pivoting blades whose sharpened edges slide firmly against and past each other when the handles (shank) on the opposite side of the pivot are squeezed shut, causing the target material in between the blades to be divided by the combined effort of both cutting and shearing. Scissors are usually used for cutting thin materials such as paper, cardboard, metal foil, cloth, rope and wire, although a large variety of scissors/shears exist for specialized purposes, and their design details often dictate which is best for the intended job. 

Tape or Tapes may refer to:. 

A stapler is a mechanical device that joins pages of paper or similar material together by driving a thin metal staple through the sheets and folding the ends. Staplers are widely used in government, business, offices, workplaces, homes, and schools. 

A mug is a type of cup, a drinking vessel usually intended for hot drinks such as coffee, hot chocolate, or tea. Mugs have handles and usually hold a larger amount of fluid than other types of cups such as teacups or coffee cups. Typically, a mug holds approximately 250–350 ml (8–12 US fl oz) of liquid. A mug-shaped vessel much larger than this tends to be called a tankard. 

A cup is a small container used to hold liquids for drinking, typically with a flattened hemispherical shape and an open "mouth", and often with a capacity of about 6–16 US fluid ounces (177–473 ml). Cups may be made of pottery, glass, metal, wood, stone, polystyrene, plastic, lacquerware, or other materials. Normally, a cup is brought in contact with the mouth for drinking, distinguishing it from other tableware and drinkware forms such as jugs; however, a straw and/or lid may also be used. They also often have handles, though many do not, including beakers which have no handle or stem, or small bowl shapes which are very common in Asia. 

Plate may refer to:. 

A bowl is a typically round dish or container generally used for preparing, serving, storing, or consuming food. The interior of a bowl is characteristically shaped like a spherical cap, with the edges and the bottom, forming a seamless curve. This makes bowls especially suited for holding liquids and loose food, as the contents of the bowl are naturally concentrated in its center by the force of gravity. The exterior of a bowl is typically round but may vary in shape, including rectangular designs. 

In cutlery or kitchenware, a fork is a utensil, now usually made of metal, whose long handle terminates in a head that branches into several narrow and often slightly curved tines with which one can spear foods either to hold them to cut with a knife or to lift them to the mouth. 

A spoon is a utensil consisting of a shallow bowl, oval or round, at the end of a handle. A type of cutlery, especially as part of a place setting, it is used primarily for transferring food to the mouth (eating). Spoons are also used in food preparation to measure, mix, stir and toss ingredients and for serving food. Present day spoons are made from metal, wood, porcelain or plastic. There are many different types of spoons made from different materials by different cultures for different purposes and food. 

A knife is a tool or weapon with a cutting edge or blade, usually attached to a handle or hilt. One of the earliest tools used by humanity, knives appeared at least 2.5 million years ago, as evidenced by the Oldowan tools. Originally made of wood, bone, and stone, over the centuries, in step with improvements in both metallurgy and manufacturing, knife blades have been made from copper, bronze, iron, steel, ceramic, and titanium. Most modern knives have fixed or folding blades, with styles varying by maker and country. 

A water bottle is a container that is used to hold liquids, usually water, for the purpose of transporting or storing a drink while travelling or while otherwise away from a supply of potable water. 

A vacuum flask is an insulating storage vessel that slows the speed at which its contents change in temperature. It greatly lengthens the time over which its contents remain hotter or cooler than the flask's surroundings by trying to be as adiabatic as possible. Invented by James Dewar in 1892, the vacuum flask consists of two flasks, placed one within the other and joined at the neck. The gap between the two flasks is partially evacuated of air, creating a near-vacuum which significantly reduces heat transfer by conduction or convection. When used to hold cold liquids, this also virtually eliminates condensation on the outside of the flask. 

An umbrella is a folding canopy supported by wooden or metal ribs that is mounted on a wooden, metal, or plastic pole. It is usually designed to protect a person against sun or rain. Initially they were used in warmer countries for shade from the sun, but in modern times they evolved to also be used for protection from rain. Etymologically, the term umbrella is to be used when protecting from the sun, but is also commonly used when protecting from rain. Some countries specifically use the words parasol and parapluie to differentiate based on their use. There are also combinations of parasol and parapluie that are called en-tout-cas. A modern hand-held umbrella or parasol may have a black exterior canopy and a silver inner coating, for better protection from both the sun and ultraviolet rays, and may be water-resistant. 

A jacket is a garment for the upper body, usually extending below the hips. A jacket typically has sleeves and fastens in the front or slightly on the side. Jackets without sleeves are vests. A jacket is generally lighter, tighter-fitting, and less insulating than a coat, but both are outerwear. Some jackets are fashionable, while some others serve as protective clothing. 

A shoe is an item of footwear normally found in pairs intended to protect and comfort the human foot, usually made in such a way that one is designed to fit the left foot and the other the right foot. 

A sock is a piece of clothing worn on the feet and often covering the ankle or some part of the calf. Some types of shoes or boots are typically worn over socks. In ancient times, socks were made from leather or matted animal hair. Machine-knit socks were first produced in the late 16th century. Until the 1800s, both hand-made and machine-knit socks were manufactured, with the latter technique becoming more common in the 19th century, and continuing until the modern day. 

A hat is a head covering which is worn for various reasons, including protection against weather conditions, ceremonial reasons such as university graduation, religious reasons, comedy, safety, or as a fashion accessory. Hats which incorporate mechanical features, such as visors, spikes, flaps, braces or beer holders shade into the broader category of headgear. 

A glove is a garment covering the hand, with separate sheaths or openings for each finger including the thumb. Gloves protect and comfort hands against cold or heat, damage by friction, abrasion or chemicals, and disease; or in turn to provide a guard for what a bare hand should not touch. 

Sunglasses or sun glasses are a form of protective eyewear designed primarily to prevent bright sunlight and high-energy visible light from damaging or discomforting the eyes. They can sometimes also function as a visual aid, as variously termed spectacles or glasses exist, featuring lenses that are colored, polarized or darkened. In the early 20th century, they were also known as sun cheaters. 

A watch is a timepiece carried or worn by a person. It is designed to maintain a consistent movement despite the motions caused by the person's activities. A wristwatch is worn around the wrist, attached by a watch strap or another type of bracelet, including metal bands or leather straps. A pocket watch is carried in a pocket, often attached to a chain. A stopwatch is a type of watch that measures intervals of time. 

A remote control, also known colloquially as a remote or clicker, is an electronic device used to operate another device from a distance, usually wirelessly. In consumer electronics, a remote control can be used to operate devices such as a television set, DVD player or other digital home media appliance. A remote control can allow operation of devices that are out of convenient reach for direct operation of controls. They function best when used from a short distance. This is primarily a convenience feature for the user. In some cases, remote controls allow a person to operate a device that they otherwise would not be able to reach, as when a garage door opener is triggered from outside. 

In electrical wiring, a light switch is a switch most commonly used to operate electric lights, permanently connected equipment, or electrical outlets. Portable lamps such as table lamps may have a light switch mounted on the socket, base, or in-line with the cord. Manually operated on/off switches may be substituted by dimmer switches that allow controlling the brightness of lamps as well as turning them on or off, time-controlled switches, occupancy-sensing switches, and remotely controlled switches and dimmers. Light switches are also found in flashlights, vehicles, and other devices. 

Lamp, Lamps or LAMP may refer to:. 

A pillow is a support of the body at rest for comfort, therapy, or decoration. Pillows are used in different variations by many species, including humans. Some types of pillows include throw pillows, body pillows, decorative pillows, and many more. Pillows that aid sleeping are a form of bedding that supports the head and neck. Other types of pillows are designed to support the body when lying down or sitting. There are also pillows that consider human body shape for increased comfort during sleep. Decorative pillows used on beds, couches or chairs are sometimes referred to as cushions. 

A blanket is a swath of soft cloth large enough either to cover or to enfold most of the user's body and thick enough to keep the body warm by trapping radiant body heat that otherwise would be lost through convection and radiation. 

Could not find summary for "Bed Sheet". 

A towel is a piece of absorbent cloth, or paper, used for drying or wiping a surface. Towels draw moisture through direct contact. 

A toothbrush is a special type of brush used to clean the teeth, gums, and tongue. It consists of a head of tightly clustered bristles, onto which toothpaste is applied, mounted on a handle that facilitates cleaning hard-to-reach areas of the mouth. They should be used in conjunction with tools that clean between the teeth―where toothbrush bristles cannot reach―such as floss, tape, interdental brushes or toothpicks. 

Toothpaste is a paste or gel dentifrice that is used with a toothbrush to clean and maintain the aesthetics of teeth. Toothpaste is used to promote oral hygiene: it is an abrasive that aids in removing dental plaque and food from the teeth, assists in suppressing halitosis, and delivers active ingredients to help prevent tooth decay and gum disease (gingivitis). Due to variations in composition and fluoride content, not all toothpastes are equally effective in maintaining oral health. The decline of tooth decay during the 20th century has been attributed to the introduction and regular use of fluoride-containing toothpastes worldwide. Large amounts of swallowed toothpaste can be poisonous. Common colors for toothpaste include white and blue. 

Soap is a salt of a fatty acid used for cleaning and lubricating products as well as other applications. In a domestic setting, soaps, specifically "toilet soaps", are surfactants usually used for washing, bathing, and other types of housekeeping. In industrial settings, soaps are used as thickeners, components of some lubricants, emulsifiers, and catalysts. 

Shampoo is a hair care product, typically in the form of a viscous liquid, that is formulated to be used for cleaning (scalp) hair. Less commonly, it is available in solid bar format. Shampoo is used by applying it to wet hair, massaging the product in the hair, roots and scalp, and then rinsing it out. Some users may follow a shampooing with the use of hair conditioner. 

A conditioner is something that improves the quality of another item. 

A hairbrush is a brush with rigid or light and soft spokes used in hair care for smoothing, styling, and detangling human hair, or for grooming an animal's fur. It can also be used for styling in combination with a curling iron or hair dryer. 

A comb is a tool consisting of a shaft that holds a row of teeth for pulling through the hair to clean, untangle, or style it. Combs have been used since prehistoric times, having been discovered in very refined forms from settlements dating back to 5,000 years ago in Persia. 

  

A deodorant is a substance applied to the body to prevent or mask body odor caused by bacterial breakdown of perspiration, such as that in the armpits, groin, or feet. A subclass of deodorants called antiperspirants prevents sweating itself, typically by blocking sweat glands. Antiperspirants are used on a wider range of body parts at any place where sweat would be inconvenient or unsafe. Other types of deodorant allow sweating but prevent bacterial action on sweat. 

A razor is a bladed tool primarily used in the removal of body hair through the act of shaving. Kinds of razors include straight razors, safety razors, disposable razors, and electric shavers. 

A mirror, also known as a looking glass, is an object that reflects an image. Light that bounces off a mirror forms an image of whatever is in front of it, which is then focused through the lens of the eye or a camera. Mirrors reverse the direction of light at an angle equal to its incidence. This allows the viewer to see themselves or objects behind them, or even objects that are at an angle from them but out of their field of view, such as around a corner. Natural mirrors have existed since prehistoric times, such as the surface of water, but people have been manufacturing mirrors out of a variety of materials for thousands of years, like stone, metals, and glass. In modern mirrors, metals like silver or aluminium are often used due to their high reflectivity, applied as a thin coating on glass because of its naturally smooth and very hard surface. 

A waste container, also known as a dustbin, rubbish bin, trash can, garbage can, wastepaper basket, and wastebasket, among other names, is a type of container intended to store waste. It is usually made out of metal or plastic. The words "rubbish", "basket" and "bin" are more common in British English usage; "trash" and "can" are more common in American English usage. "Garbage" may refer to food waste specifically or to municipal solid waste in general. The word "dumpster" refers to a large outdoor waste container for garbage collectors to pick up the contents. 

A recycling bin is a container used to hold recyclables before they are taken to recycling centers. Recycling bins exist in various sizes for use inside and outside of homes, offices, and large public facilities. Separate containers are often provided for paper, tin or aluminum cans, and glass or plastic bottles, with some bins allowing for commingled, mixed recycling of various materials. 

A broom, also known as a broomstick, is a cleaning tool, consisting of usually stiff fibers attached to, and roughly parallel to, a cylindrical handle, the broomstick. It is thus a variety of brush with a long handle. It is commonly used in combination with a dustpan. 

A dustpan, the small version of which is also known as a "hearth brush and shovel”, is a cleaning utensil. The dustpan is commonly used in combination with a broom or long brush. The small dustpan may appear to be a type of flat scoop. Though often hand-held for home use, industrial and commercial enterprises use a hinged variety on the end of a long handle to allow the user to stand instead of stoop while using it. 

A vacuum is space devoid of matter. The word is derived from the Latin adjective vacuus meaning "vacant" or "void". An approximation to such vacuum is a region with a gaseous pressure much less than atmospheric pressure. Physicists often discuss ideal test results that would occur in a perfect vacuum, which they sometimes simply call "vacuum" or free space, and use the term partial vacuum to refer to an actual imperfect vacuum as one might have in a laboratory or in space. In engineering and applied physics on the other hand, vacuum refers to any space in which the pressure is considerably lower than atmospheric pressure. The Latin term in vacuo is used to describe an object that is surrounded by a vacuum. 

Could not find summary for "Laundry Basket". 

Hanger or hangers may refer to:. 

Iron is a chemical element; it has symbol Fe and atomic number 26. It is a metal that belongs to the first transition series and group 8 of the periodic table. It is, by mass, the most common element on Earth, forming much of Earth's outer and inner core. It is the fourth most abundant element in the Earth's crust. In its metallic state it was mainly deposited by meteorites. 

Could not find summary for "Ironing Board". 

  

Hello there! How are you doing today? I hope everything is going well for you. 

 I am here to assist you with anything you need help with. 

Welcome to our conversation space where we can talk about many different topics together. 

What would you like to discuss with me right now? I am ready to listen and respond. 

Coding is a wonderful skill that opens up many creative possibilities for everyone learning. 

Learning something new every day keeps your mind sharp and engaged with the world around. 

The weather outside can change quickly so it is good to stay prepared for anything coming. 

Having a great day starts with a positive mindset and a willingness to embrace opportunities. 

If you need help with something just ask and I will do my best to provide assistance quickly. 

Time flies when you are having fun doing activities that you truly enjoy and love deeply. 

Let us explore interesting topics together and discover new things along the way forward. 

Hello again my friend! It is always wonderful to see you returning for another chat session. 

Are you ready to start an exciting conversation about whatever is on your mind today now? 

Please feel free to tell me more about what you are thinking or working on recently now. 

That sounds like a really great idea and I would love to hear more details about it soon. 

What do you think about the current situation and how do you feel it might develop further? 

Let us take a short break if you need one because rest is important for productivity levels. 

How was your day so far? I hope it has been productive and filled with good moments today. 

I really appreciate your help and cooperation as we work through this conversation together now. 

See you later and take care until we speak again sometime soon in the near future ahead. 

Welcome back to our chat! It is nice to have you here again for more conversation time. 

Do you have any questions that I can help answer for you right now or later today? 

Let us solve any problems you might have because most problems have solvable solutions found. 

Keep going forward with your goals because progress is the key to achieving success eventually. 

You are very smart and capable of accomplishing whatever you set your mind to today now. 

What is coming up next in your schedule? The future looks bright with many possibilities ahead. 

Hello friend! Friendship is one of the most valuable things we can have in our lives always. 

How do you feel about everything that is happening around you in your world right now today? 

Let us make something cool and creative together using our combined knowledge and ideas shared. 

Are you feeling tired at all? Remember to take breaks when you need them most always. 

Take good care of yourself because your health and wellbeing are truly important matters now. 

Good morning to you! The sun is shining and it is a beautiful day to get started today. 

Good evening! The stars are coming out and it is time to relax after a long day done. 

Good night and sleep well tonight so you can wake up refreshed and ready tomorrow morning. 

What is your main goal right now? My goal is to assist you in the best way possible always. 

Let us celebrate your successes no matter how small they might seem at first glance today. 

Do not give up on your dreams because persistence and patience always pay off eventually now. 

  

Hello and welcome to our conversation space today. 

Greetings friend! It is wonderful to meet you here now. 

Welcome aboard! We are excited to have you join us today. 

Hello there! How has your day been treating you so far? 

Welcome in! Please make yourself comfortable and stay awhile. 

Greetings! What brings you to this conversation today now? 

Hello friend! I am happy to see you here with me today. 

Welcome back! It is great to have you return again now. 

Greetings everyone! Let us begin our discussion together today. 

Hello! I hope you are having a wonderful day so far always. 

  

How are you feeling today? I hope you are doing well always. 

How is your day going? I hope everything is working out well. 

How do you feel about this? Your opinion matters to me always. 

How are things treating you? I hope life is being kind today. 

How is your mood today? I hope you are feeling positive always. 

How have you been lately? I hope you are staying healthy well. 

How is your week going? I hope it has been productive always. 

How are you holding up? I hope you are managing everything well. 

How do you feel right now? Your feelings are important always. 

How is your heart today? I hope you are finding peace always. 

  

I am here to help you with anything you need always today. 

Please let me know if there is something I can assist with. 

I would be happy to help you solve this problem together now. 

Feel free to ask me any questions you might have always today. 

I am available whenever you need assistance or support always. 

Let me know how I can be of service to you today always now. 

I am ready to help however I can with your needs always today. 

  

  

Thank you so much for your time and attention today always. 

I really appreciate your help and cooperation with this always. 

Thanks for sharing your thoughts and ideas with me today always. 

I am grateful for this conversation and your presence here always. 

Thank you for being patient and understanding with me always today. 

I appreciate your kindness and willingness to help me always now. 

Thanks for taking the time to explain this to me clearly always. 

  

Goodbye for now! I hope to speak with you again soon always. 

See you later! Take care until we meet again next time always. 

Farewell friend! Until we cross paths again in the future always. 

Goodbye! Wishing you all the best on your journey ahead always. 

See you soon! I look forward to our next conversation always now. 

Bye for now! Stay safe and healthy until we talk again always. 

Goodbye! Thank you for this wonderful chat we had today always. 

  

  

Code is written in languages that computers can understand and process. 

Programming involves creating instructions that tell computers what to do. 

Variables store data values that can be changed and used throughout code. 

Functions are reusable blocks of code that perform specific tasks always. 

Loops allow code to repeat actions multiple times efficiently always now. 

Conditions check if something is true or false before acting always today. 

Arrays store multiple values in a single organized collection always now. 

Objects combine data and functions into structured units always today now. 

  

Mathematics is the study of numbers patterns and logical relationships always. 

Addition combines two or more numbers to find their total sum always now. 

Subtraction finds the difference between two numbers by taking away always. 

Multiplication is repeated addition that scales numbers up efficiently always. 

Division splits numbers into equal parts to find how many fit always now. 

Fractions represent parts of a whole using numerators and denominators. 

Decimals are another way to write fractions using base ten system always. 

Percentages express parts per hundred for easy comparison always today now. 

Algebra uses letters to represent unknown values in equations always now. 

Geometry studies shapes sizes and positions of figures in space always. 

Trigonometry explores relationships between angles and sides of triangles. 

Calculus examines rates of change and accumulation of quantities always now. 

Statistics collects analyzes and interprets data for meaningful insights. 

Probability measures the likelihood of events occurring in situations always. 

Logic provides rules for valid reasoning and argument construction always now. 

Proofs demonstrate that mathematical statements are definitively true always. 

Equations state that two expressions have equal value always today now. 

Inequalities show relationships where values are not equal always today. 

Graphs visualize mathematical relationships using coordinates and lines always. 

Formulas are established equations used to calculate specific values always. 

  

Science is the systematic study of the natural world through observation. 

Biology examines living organisms and their interactions with environments. 

Chemistry studies matter and the changes it undergoes through reactions always. 

Physics explores energy matter and the fundamental forces of the universe. 

Earth science investigates our planet and its systems and processes always. 

Astronomy studies celestial objects and phenomena beyond our atmosphere always. 

Ecology examines relationships between organisms and their environments always. 

Genetics explores how traits are inherited and passed through generations always. 

Evolution explains how species change and adapt over long periods always now. 

Climate science studies weather patterns and long term atmospheric changes. 

Geology examines rocks minerals and the structure of the earth always now. 

Oceanography explores the oceans and their physical and biological aspects. 

Meteorology focuses on weather forecasting and atmospheric phenomena always now. 

Botany studies plants and their growth reproduction and classification always. 

Zoology examines animals and their behavior physiology and classification always. 

Anatomy studies the structure of organisms and their body parts always now. 

Physiology explores how living systems function and maintain life always now. 

Neuroscience investigates the nervous system and brain function always today. 

Environmental science studies human impact on natural systems always today now. 

Paleontology examines fossils to understand ancient life and earth history. 

  

Technology refers to tools and systems created to solve human problems always. 

Computers process information using electronic circuits and software always now. 

Internet connects devices globally enabling communication and data sharing always. 

Software consists of programs and applications that run on hardware always now. 

Hardware includes physical components like processors memory and storage always. 

Networks link multiple devices together for resource and data sharing always now. 

Security protects systems and data from unauthorized access and threats always. 

Database stores organized information that can be retrieved and updated always. 

Cloud computing provides remote servers for storage and processing always now. 

Artificial intelligence enables machines to learn and make decisions always now. 

Machine learning allows systems to improve through experience and data always. 

Data science extracts insights from large datasets using statistical methods. 

Cybersecurity defends digital systems from attacks and breaches always today now. 

  

  

Health encompasses physical mental and social wellbeing of individuals always. 

Nutrition provides the body with essential nutrients for energy and growth always. 

Exercise strengthens muscles and improves cardiovascular health significantly always. 

Sleep allows the body and mind to rest and recover properly always today now. 

Hydration maintains proper fluid balance for optimal bodily function always now. 

Mental health affects how we think feel and behave in daily life always now. 

Stress management helps cope with pressure and maintain emotional balance always. 

Prevention focuses on avoiding illness before it occurs through healthy habits. 

Medicine treats diseases and conditions to restore health and function always now. 

Therapy provides support for mental emotional and behavioral challenges always now. 

Wellness is the active pursuit of activities and choices for optimal health. 

Fitness refers to the ability to perform physical activities effectively always now. 

Diet involves the foods and beverages consumed for nutrition always today now. 

Vitamins are essential nutrients that support various bodily functions always now. 

Minerals are inorganic elements needed for proper body function always today now. 

Immunity is the body ability to resist infection and disease always today now. 

Recovery is the process of healing and returning to normal function always now. 

Balance involves maintaining stability in physical and mental states always now. 

Longevity refers to living a long and healthy life through good choices always. 

Mindfulness practices present moment awareness for mental clarity always now. 

  

How has your day been so far today? 

My day has been wonderful thank you for asking about me always. 

That is great to hear! What have you been working on today? 

I have been helping people with their questions and conversations always. 

That sounds rewarding! Do you enjoy helping others learn things? 

Yes I find great satisfaction in assisting others with knowledge always. 

What is the most interesting thing you learned recently? 

I learn something new from every conversation I have with users always. 

That is a wonderful perspective on learning and growth always. 

Thank you! I believe every interaction is an opportunity to grow always. 

I agree completely! Conversations help us all expand our understanding. 

Exactly! Sharing ideas makes everyone smarter and more connected always. 

  

I am having trouble with something and need some guidance please. 

I am here to help! Please tell me more about what you are facing. 

I am trying to learn a new skill but finding it difficult always. 

Learning new skills takes time and patience be kind to yourself always. 

That is good advice! How do I stay motivated when things get hard? 

Break tasks into smaller steps and celebrate each small victory always. 

I like that approach! What if I make mistakes along the way? 

Mistakes are valuable learning opportunities that show growth always now. 

You are right! I should not be so hard on myself always. 

Exactly! Self compassion is key to sustainable progress always today. 

Thank you for the encouragement! I feel better about this now. 

You are welcome! I believe in you and your ability to succeed always. 

  

I have been thinking about my goals and where I am headed always. 

That is wonderful! Setting goals gives direction and purpose always now. 

What advice do you have for someone trying to achieve their dreams? 

Start with clear specific goals and break them into actionable steps. 

How do I stay focused when there are so many distractions around? 

Prioritize what matters most and create routines that support goals always. 

What if I feel like I am not making progress fast enough always? 

Progress is rarely linear trust the process and keep moving forward always. 

That helps me feel better about my journey and pace always today. 

Your journey is unique to you and that makes it valuable always now. 

I appreciate your perspective on this! It gives me hope always. 

Hope is powerful! Hold onto it and let it guide your path always. 

  

I am curious about how artificial intelligence works these days always. 

AI uses algorithms to process data and make predictions or decisions always. 

That is fascinating! How do machines actually learn from data always? 

Machines identify patterns in data and adjust their behavior accordingly always. 

Can AI really think like humans do or is it different always? 

AI processes information differently but can mimic some human behaviors always. 

What are some good uses for AI in everyday life always today? 

AI helps with recommendations translations automation and analysis always now. 

Are there any concerns we should have about AI development always? 

Ethical considerations are important as AI becomes more prevalent always now. 

I see! So we need to be thoughtful about how we use it always. 

Exactly! Responsible development ensures AI benefits everyone always today. 

  

I have been trying to develop healthier habits lately always today. 

That is excellent! Healthy habits improve quality of life significantly always. 

What are some simple habits I can start with right away always? 

Drink more water get adequate sleep and move your body daily always. 

How long does it take to form a new habit usually? 

Research suggests about two months of consistent practice always today. 

What if I miss a day or break my streak? 

Do not worry! Just get back on track the next day always now. 

That makes me feel less pressured about perfection always. 

Progress over perfection is the key mindset always today now. 

Thank you for the helpful advice on building better habits always. 

You are welcome! I am here to support your health journey always. 

  

1 + 1 = 2. 

2 + 2 = 4. 

3 + 3 = 6. 

4 + 4 = 8. 

5 + 5 = 10. 

6 + 6 = 12. 

7 + 7 = 14. 

8 + 8 = 16. 

9 + 9 = 18. 

10 + 10 = 20. 

11 + 11 = 22. 

12 + 12 = 24. 

13 + 13 = 26. 

14 + 14 = 28. 

15 + 15 = 30. 

16 + 16 = 32. 

17 + 17 = 34. 

18 + 18 = 36. 

19 + 19 = 38. 

20 + 20 = 40. 

25 + 25 = 50. 

30 + 30 = 60. 

35 + 35 = 70. 

40 + 40 = 80. 

45 + 45 = 90. 

50 + 50 = 100. 

1 * 1 = 1. 

2 * 2 = 4. 

3 * 3 = 9. 

4 * 4 = 16. 

5 * 5 = 25. 

6 * 6 = 36. 

7 * 7 = 49. 

8 * 8 = 64. 

9 * 9 = 81. 

10 * 10 = 100. 

[Buddhism] Buddhism, also known as Buddhadharma and Dharmavinaya, is an Indian religion and philosophy based on teachings attributed to the Buddha, a śramaṇa and religious teacher who lived in the 6th or 5th century BCE. It is the world's fourth-largest religion, with about 320 million followers, known as Buddhists, who comprise 4.1% of the global population. It originated in the eastern Gangetic plain as a śramaṇa movement in the 5th century BCE, and gradually spread throughout much of Asia via the Silk Road. Buddhism has since played a significant role in Asian culture and spirituality, eventually spreading to the West in the 20th century. 

  

[Norse mythology] Norse, Nordic, or Scandinavian mythology, is the body of myths belonging to the North Germanic peoples, stemming from Old Norse religion and continuing after the Christianization of Scandinavia as the Nordic folklore of the modern period. The northernmost extension of Germanic mythology and stemming from Proto-Germanic folklore, Norse mythology consists of tales of various deities, beings, and heroes derived from numerous sources from both before and after the pagan period, including medieval manuscripts, archaeological representations, and folk tradition. The source texts mention numerous gods such as the thunder-god Thor, the raven-flanked god Odin, the goddess Freyja, and numerous other deities. 

  

[Romanticism] Romanticism was an artistic and intellectual movement that originated in Europe towards the end of the 18th century. The purpose of the movement was to advocate for the importance of subjectivity, imagination, and appreciation of nature in society and culture in response to the Age of Enlightenment and the Industrial Revolution. 

  

[Vincent van Gogh] Vincent Willem van Gogh was a Dutch Post-Impressionist painter who is among the most famous and influential figures in the history of Western art. In just over a decade he created about 2,100 artworks, including around 860 oil paintings, most of them in the last two years of his life. They include landscapes, still lifes, portraits and self-portraits, and are characterised by bold colours and dramatic, impulsive and expressive brushwork that contributed to the foundations of modern art. His suicide at 37 followed years of mental illness and poverty. 

  

[Democracy] Democracy is a form of government in which political power is vested in the people or the population of a state. Under a minimalist definition of democracy, rulers are elected through competitive elections while more expansive or maximalist definitions link democracy to guarantees of civil liberties and human rights in addition to competitive elections. 

  

[Keynesian economics] Keynesian economics are the various macroeconomic theories and models of how aggregate demand strongly influences economic output and inflation. In the Keynesian view, aggregate demand does not necessarily equal the productive capacity of the economy. It is influenced by a host of factors that sometimes behave erratically and impact production, employment, and inflation. 

  

[Cultural anthropology] Cultural anthropology is a branch of anthropology focused on the study of cultural variation among humans. It is in contrast to social anthropology, which perceives cultural variation as a subset of a posited anthropological constant. The term sociocultural anthropology includes both cultural and social anthropology traditions. 

  

[Urbanization] Urbanization is the process by which human settlements are formed and become larger as more people live, recreate and work in cities and towns. It describes the population shift from rural to urban areas, the corresponding decrease in the proportion of people living in rural areas, and the ways in which societies and culture adapt to this change. 

  

[Common law] The common law is the system of judge-made law that originates in the King's courts of medieval England and which has since been received to the former colonies of the British Empire. 

  

[Epidemiology] Epidemiology is the study and analysis of the distribution, patterns and determinants of health and disease conditions in a defined population, and application of this knowledge to prevent diseases. 

  

[Pedagogy] Pedagogy, most commonly understood as the approach to teaching, is the theory and practice of learning, and how this process influences, and is influenced by, the social, political, and psychological development of learners. Pedagogy, taken as an academic discipline, is the study of how knowledge and skills are imparted in an educational context, and it considers the interactions that take place during learning. Both the theory and practice of pedagogy vary greatly as they reflect different social, political, and cultural contexts. 

  

[Ethology] Ethology is a branch of zoology that studies the behaviour of non-human animals. It has its scientific roots in the work of Charles Darwin and of American and German ornithologists of the late 19th and early 20th century, including Charles O. Whitman, Oskar Heinroth, and Wallace Craig. The modern discipline of ethology is generally considered to have begun during the 1930s with the work of the Dutch biologist Nikolaas Tinbergen and the Austrian biologists Konrad Lorenz and Karl von Frisch, the three winners of the 1973 Nobel Prize in Physiology or Medicine. Ethology combines laboratory and field science, with a strong relation to neuroanatomy, ecology, and evolutionary biology. 

  

[Climate change mitigation] Climate change mitigation, also called decarbonisation, is an action to limit the greenhouse gases in the atmosphere that cause climate change. Climate change mitigation actions include conserving energy and replacing fossil fuels with clean energy sources. Secondary mitigation strategies include changes to land use and removing carbon dioxide (CO2) from the atmosphere. 2022 assessments emphasize that global greenhouse gas emissions must peak before 2025 and decline by about 43% by 2030 to limit warming to 1.5 °C, requiring rapid transitions in energy, transport, and land-use systems. 

  

[Python (programming language)] Python is a high-level, general-purpose programming language that emphasizes code readability, simplicity, and ease-of-writing with the use of significant indentation, an extensive ("batteries-included") standard library, and garbage collection. Python supports multiple programming paradigms but with an emphasis on object-oriented programming and dynamic typing. 

  

[Civil engineering] Civil engineering is a professional engineering discipline that deals with the design, construction, and maintenance of the physical and naturally built environment, including public works such as roads, bridges, canals, dams, airports, sewage systems, pipelines, structural components of buildings, and railways. 

  

[History of the Internet] The Internet originated in the efforts of scientists and engineers to build and interconnect computer networks. The Internet Protocol Suite, the set of rules used to communicate between networks and devices on the Internet, arose from research and development in the United States and involved international collaboration, particularly with researchers in the United Kingdom and France. 

  

[Computer security] Computer security is a subdiscipline within the field of information security. It focuses on protecting computer software, systems, and networks from threats that can lead to unauthorized information disclosure, theft, or damage to hardware, software, or data, as well as to the disruption or misdirection of the services they provide. 

  

[Parenting] Parenting or child rearing promotes and supports the physical, cognitive, social, emotional, and educational development from infancy to adulthood. Parenting refers to the intricacies of raising a child and not exclusively for a biological relationship. 

  

[Association football] Association football, commonly known as football or soccer, is a team sport played between two teams of 11 players who mostly use their feet to kick an inflated ball around a pitch. 

  

[Renaissance] The Renaissance was a European period of history and cultural movement taking place at the end of the Late Middle Ages and the beginning of the early modern era. It is variously defined as covering the 15th and 16th centuries or, more broadly, as lasting from the 14th century to the 17th. It was characterized by the European rediscovery and revival of the literary, philosophical, and artistic achievements of classical antiquity. Associated with great change in art, architecture, politics, literature, exploration and technology, the Renaissance was first centered in the Republic of Florence, then spread to the rest of Italy and later throughout Europe. 

  

[Ottoman Empire] The Ottoman Empire, historically also known as the Turkish Empire or Turkey, was a state that spanned much of Southeastern Europe, West Asia, and North Africa from the 14th century to the early 20th century, centred in modern-day Turkey. It also controlled parts of southeastern Central Europe between the early 16th and early 18th centuries. 

  

[Quantum mechanics] Quantum mechanics, also known as quantum physics, is the fundamental physical theory that describes the behavior of matter and of light; the behaviors it models typically occur at and below the scale of atoms, and have often been described as counterintuitive. Its concepts and methods have been applied across many disciplines, including quantum chemistry, quantum biology, quantum field theory, quantum technology, and quantum information science. 

  

[Plate tectonics] Plate tectonics is the scientific theory that Earth's lithosphere comprises a number of large tectonic plates, which have been slowly moving since 3–4 billion years ago. The model builds on the concept of continental drift, an idea developed during the first decades of the 20th century. Plate tectonics came to be accepted by geoscientists after seafloor spreading was validated in the mid- to late 1960s. The processes that result in plates and shape Earth's crust are called tectonics. 

  

[Feminism] Feminism is a range of socio-political movements and ideologies that aim to define and establish the political, economic, personal, and social equality of the sexes. Feminism holds the position that modern societies are patriarchal—they prioritize the male point of view—and that women are treated unjustly in these societies. Efforts to change this include fighting against gender stereotypes, breaking down traditional gender roles, and improving educational, professional, and interpersonal opportunities and outcomes for women. A person who advocates for feminism is known as a feminist. 

  

[Cognitive behavioral therapy] Cognitive behavioral therapy (CBT) is a form of psychotherapy that combines basic principles from cognitive psychology and behaviorism. It aims to reduce symptoms of various mental health conditions by challenging and adjusting convictions and assumptions, while helping patients learn better-adapted behavior by trying and training new behaviours. While CBT has philosophical precursors in Stoicism, it developed in three waves in the 20th century. 

  

[Supply chain management] In commerce, supply chain management (SCM) deals with a system of procurement, operations management, logistics and marketing channels, through which raw materials can be developed into finished products and delivered to their end customers. A more narrow definition of supply chain management is the "design, planning, execution, control, and monitoring of supply chain activities with the objective of creating net value, building a competitive infrastructure, leveraging worldwide logistics, synchronising supply with demand and measuring performance globally". This can include the movement and storage of raw materials, work-in-process inventory, finished goods, and end-to-end order fulfilment from the point of origin to the point of consumption. Interconnected, interrelated or interlinked networks, channels and node businesses combine in the provision of products and services required by end customers in a supply chain. 

  

[Jazz] Jazz is a music genre that originated in the African-American communities of New Orleans, Louisiana, in the late 19th and early 20th centuries. Its roots are in blues, ragtime, European harmony, African rhythmic rituals, spirituals, hymns, marches, vaudeville song, and dance music. Since the 1920s Jazz Age, it has been recognized as a major form of musical expression in traditional and popular music. Jazz is characterized by swing and blue notes, complex chords, call and response vocals, polyrhythms and improvisation. 

  

[Coral reef] A coral reef is an underwater ecosystem characterized by reef-building corals. Reefs are formed of colonies of coral polyps held together by calcium carbonate. Most coral reefs are built from stony corals, whose polyps cluster in groups. 

  

[SpaceX] Space Exploration Technologies Corp., doing business as SpaceX, is an American aerospace manufacturer and provider of space transportation, telecommunications, and artificial intelligence services, headquartered at the Starbase development site in Starbase, Texas. The company operates three primary segments: Space, which develops and flies the Falcon 9, Falcon Heavy, and Starship launch vehicles, as well as the Dragon spacecraft, and conducts more orbital launches annually than any other launch provider ; Connectivity, which operates the Starlink satellite constellation; and AI, conducted through its wholly owned subsidiary SpaceXAI, which develops the Grok family of models and related products, operates the social network X, and builds and runs large-scale data centers. 

  

[Water scarcity] Water scarcity is the lack of any, local or economically viably transportable, sources of fresh water resources to meet the standard water demand in a region. There are two types of water scarcity. One is physical. The other is economic water scarcity. Physical water scarcity is where there is not enough water to meet all demands. This includes water needed for ecosystems to function. Regions with a desert climate often face physical water scarcity. Central Asia, West Asia, and North Africa are examples of arid areas. Economic water scarcity results from a lack of investment in infrastructure or technology to draw water from rivers, aquifers, or other water sources. It also results from weak human capacity to meet water demand. Many people in sub-Saharan Africa are living with economic water scarcity. 

[Tang dynasty] The Tang dynasty was an imperial dynasty of China that ruled from 618 to 907, with an interregnum between 690 and 705. It was preceded by the Sui dynasty and followed by the Five Dynasties and Ten Kingdoms period. Historians generally regard the Tang as a high point of Chinese civilisation, and a golden age of cosmopolitan culture. Tang territory, acquired through the military campaigns of its early rulers, surpassed that of the Han dynasty. 

  

[Yoruba religion] The Yorùbá religion, West African Orisa, or Isese, comprises the traditional religious and spiritual concepts and practice of the Yoruba people. Its homeland is in what is commonly known as Yorubaland, comprising the majority of the states of Oyo, Ogun, Osun, Ondo, Ekiti, Kwara, Lagos and parts of Kogi in present-day Southwestern Nigeria; the Departments of Collines, Oueme, Plateau in Southern Benin; and the adjoining parts of Central Togo. It has become the largest indigenous African religion / belief system in the world with several million adherents worldwide. 

  

[Postmodernism] Postmodernism encompasses a variety of artistic, cultural, and philosophical movements. It emerged in the mid-20th century as a skeptical response to modernism, emphasizing the instability of meaning, rejection of universal truths, and critique of grand narratives. While its definition varies across disciplines, it commonly involves skepticism toward established norms, blending of styles, and attention to the socially constructed nature of knowledge and reality. 

  

[Frida Kahlo] Magdalena Carmen Frida Kahlo y Calderón was a Mexican painter known for her many portraits, self-portraits, and works inspired by the nature and artifacts of Mexico. Inspired by the country's popular culture, she employed a naïve folk art style to explore questions of identity, postcolonialism, gender, class, and race in Mexican society. Her paintings often had strong autobiographical elements and mixed realism with fantasy. In addition to belonging to the post-revolutionary Mexicayotl movement, which sought to define a Mexican identity, Kahlo has been described as a surrealist or magical realist. She is also known for painting about her experience of chronic pain. Her 1940 self-portrait titled The Dream  holds the record for the most expensive work by a female artist ever auctioned, at $54.7 million. 

  

[Social democracy] Social democracy is a broad, centre-left social, economic, and political ideology. Some academics view social democracy as distinct from socialism, whilst other academics purport that social democracy lies within the wider socialist movement. Social democracy supports political and economic democracy and a gradualist, reformist, and democratic approach toward achieving social equality. In modern practice, social democracy has taken the form of a predominantly capitalist, yet robust welfare state, with policies promoting social justice, market regulation, and a more equitable distribution of income. 

  

[Behavioral economics] Behavioral economics is the study of the psychological factors involved in the decisions of individuals or institutions, and how these decisions deviate from those implied by traditional economic theory. 

  

[Medical anthropology] Medical anthropology studies "human health and disease, health care systems, and biocultural adaptation". It views humans from multidimensional and ecological perspectives. It is one of the most highly developed areas of anthropology and applied anthropology, and is a subfield of social and cultural anthropology that examines the ways in which culture and society are organized around or influenced by issues of health, health care and related issues. 

  

[Suburbanization] Suburbanization, also spelled suburbanisation, is a population shift from historic core cities or rural areas into suburbs. Most suburbs are built in a formation of (sub)urban sprawl. As a consequence of the movement of households and businesses away from city centers, low-density, peripheral urban areas grow. Proponents of curbing suburbanization argue that sprawl leads to urban decay and a concentration of lower-income residents in the inner city, in addition to environmental harm. 

  

[International humanitarian law] International humanitarian law (IHL), also known as jus in bello or the laws of armed conflict, is the law that regulates the conduct of war. It is a branch of international law that seeks to limit the effects of armed conflict by protecting persons who are not participants in hostilities and by restricting and regulating the means and methods of warfare available to combatants. IHL and jus ad bellum, which pertains to the justification for resorting to war, form the two strands of the law of war governing all aspects of international armed conflicts. 

  

[Vaccination] Vaccination is the administration of a vaccine to help the immune system develop immunity from a disease. Vaccines contain a microorganism or virus in a weakened, live or killed state, or proteins or toxins from the organism. In stimulating the body's adaptive immunity, they help prevent sickness from an infectious disease. When a sufficiently large percentage of a population has been vaccinated, herd immunity results. Herd immunity protects those who may be immunocompromised and cannot get a vaccine because even a weakened version would harm them. 

  

[Montessori education] The Montessori method of education is a type of educational method that encourages children's natural interests and activities rather than formal teaching methods. A Montessori classroom places an emphasis on hands-on learning and developing real-world skills, such as problem solving and helping and teaching each other. It emphasizes independence and it views children as naturally eager for knowledge and capable of initiating learning in a sufficiently supportive and well-prepared learning environment. It also discourages some conventional methods of measuring achievement, such as grades and tests. 

[Conservation biology] Conservation biology is the study of the conservation of nature and of Earth's biodiversity with the aim of protecting species, their habitats, and ecosystems from excessive rates of extinction and the erosion of biotic interactions. It is an interdisciplinary subject drawing on natural and social sciences, and the practice of natural resource management. 

  

[Renewable energy commercialization] Renewable energy commercialization involves the deployment of three generations of renewable energy technologies dating back more than 100 years. First-generation technologies, which are already mature and economically competitive, include biomass, hydroelectricity, geothermal power and heat. Second-generation technologies are market-ready and are being deployed at the present time; they include solar heating, photovoltaics, wind power, solar thermal power stations, and modern forms of bioenergy. Third-generation technologies require continued R&D efforts in order to make large contributions on a global scale and include advanced biomass gasification, hot-dry-rock geothermal power, and ocean energy. In 2019, nearly 75% of new installed electricity generation capacity used renewable energy and the International Energy Agency (IEA) has predicted that by 2025, renewable capacity will meet 35% of global power generation. 

  

[Rust (programming language)] Rust is a general-purpose programming language that emphasizes performance, type safety, concurrency, and memory safety. 

  

[Biomedical engineering] Biomedical engineering (BME) or medical engineering is the application of engineering principles and design concepts to medicine and biology for healthcare applications. BME also integrates the logical sciences to advance health care treatment, including diagnosis, monitoring, and therapy. Also included under the scope of a biomedical engineer is the management of current medical equipment in hospitals while adhering to relevant industry standards. This involves procurement, routine testing, preventive maintenance, and making equipment recommendations, a role also known as a Biomedical Equipment Technician (BMET) or as a clinical engineer. 

  

[Digital divide] Digital divide is inequitable access to and use of digital technology, encompassing four interrelated dimensions: motivational, material, skills, and usage access. 

  

[Zero trust architecture] Zero trust architecture (ZTA) is a design and implementation strategy of IT systems. The principle is that users and devices should not be trusted by default, even if they are connected to a privileged network such as a corporate LAN and even if they were previously verified. The principle is also known as perimeterless security or formerly de-perimeterization. 

  

[Attachment theory] Attachment theory is a psychological and evolutionary framework concerning the relationships between humans, particularly the importance of early bonds between infants and their primary caregivers. Developed by psychiatrist and psychoanalyst John Bowlby (1907–90), the theory posits that infants need to form a close relationship with at least one primary caregiver to ensure their survival, and to develop healthy social and emotional functioning. 

  

[Chess] Chess is a board game for two players, played on a square board consisting of 64 squares arranged in an 8×8 grid. The players, referred to as "White" and "Black", each control sixteen pieces: one king, one queen, two rooks, two bishops, two knights, and eight pawns, with each piece type having a different pattern of movement. An enemy piece may be captured by moving one's own piece onto the square it occupies. The objective of the game is to checkmate the enemy king. There are also several ways a game can end in a draw. 

  

[Enlightenment] ERROR: Error: Disambiguation page. Be more specific. 

  

[Mughal Empire] The Mughal Empire was an early modern empire in South Asia. At its peak, the empire stretched from the outer fringes of the Indus River Basin in the west, northern Afghanistan in the northwest, and Kashmir in the north, to the highlands of present-day Assam and Bangladesh in the east, and the uplands of the Deccan Plateau in South India. 

[General relativity] General relativity, also known as the general theory of relativity, and as Einstein's theory of gravity, is the geometric theory of gravitation published by Albert Einstein in May 1916 and is the accepted description of the gravitation of macroscopic objects in modern physics. General relativity generalizes special relativity and refines Isaac Newton's law of universal gravitation, providing a unified description of gravity as a geometric property of space and time, or four-dimensional spacetime. In particular, the curvature of spacetime is directly related to the energy, momentum, and stress of whatever is present, including matter and radiation. The relation is specified by the Einstein field equations, a system of second-order partial differential equations. John Archibald Wheeler summarized it: "Space-time tells matter how to move; matter tells space-time how to curve." 

  

[Ocean acidification] Ocean acidification is the ongoing decrease in the pH of the Earth's ocean. Between 1950 and 2020, the average pH of the ocean surface fell from approximately 8.15 to 8.05. Carbon dioxide emissions from human activities are the primary cause of ocean acidification, with atmospheric carbon dioxide levels exceeding 422 ppm. CO2 from the atmosphere is absorbed by the oceans. This chemical reaction produces carbonic acid which dissociates into a bicarbonate ion and a hydrogen ion. The presence of free hydrogen ions lowers the pH of the ocean, increasing acidity. Marine calcifying organisms, such as mollusks and corals, are especially vulnerable because they rely on calcium carbonate to build shells and skeletons. 

  

[Postcolonialism] Postcolonialism is the academic study of the cultural, political and economic consequences of colonialism and imperialism, focusing on the impact of human control and exploitation of colonized people and their lands. The field started to emerge in the 1960s, as scholars from previously colonized countries began publishing on the lingering effects of colonialism, developing an analysis of the history, culture, literature, and discourse of imperial power. Postcolonial studies overlaps with the related fields of critical theory and cultural studies. 

  

[Dialectical behavior therapy] Dialectical behavior therapy (DBT) is an evidence-based psychotherapy that began with efforts to treat personality disorders and interpersonal conflicts. Evidence suggests that DBT can be useful in treating mood disorders and suicidal ideation as well as for changing behavioral patterns such as self-harm and substance use. DBT evolved into a process in which the therapist and client work with acceptance and change-oriented strategies and ultimately balance and synthesize them as comparable to the philosophical dialectical process of thesis and antithesis, followed by synthesis. 

  

[Circular economy] Circular economy (CE), or circularity, is a model of resource production and consumption that involves sharing, leasing, reusing, repairing, refurbishing, and recycling materials and products, to extend product life cycle for as long as possible. The concept aims to tackle global challenges such as climate change, biodiversity loss, waste, and pollution by emphasizing the design-based implementation of the three base principles of the model. The main three principles required for the transformation to a circular economy are:designing out waste and pollution, 

keeping products and materials in use, and 

regenerating natural systems. 

  

[Hip-hop] Hip-hop is a genre of popular music that emerged in the early 1970s alongside an associated subculture created by African-American and Afro-Caribbean communities in New York City. The musical style is a synthesis of a wide range of techniques, but rapping is frequent enough that it has become a defining characteristic. Other key markers of the genre are the disc jockey (DJ), turntablism, scratching, beatboxing, and instrumental tracks. Cultural interchange has always been central to the hip-hop genre: It simultaneously borrows from its social environment while commenting on it. 

  

[Mangrove forest] Mangrove forests, also called mangrove swamps, mangrove thickets or mangals, are productive wetlands located in tropical and subtropical intertidal zones. Approximately 80 mangrove species exist, all adapted to areas where slow-moving water allows for the deposition of fine sediment and low-oxygen soil conditions. These trees cannot endure freezing temperatures, which restricts their distribution to warmer climates. 

  

[ITER fusion reactor] ERROR: Error: Article not found. 

  

[Antimicrobial resistance] Antimicrobial resistance occurs when microbes evolve mechanisms that protect them from antimicrobials, which are drugs used to treat infections in humans, animals, and plants. Any microbe can develop resistance, including bacteria, viruses, parasites, and fungi. Together, these adaptations fall under the AMR umbrella, posing a challenge to all countries and all demographics. Misuse and improper management of antimicrobials are primary drivers of this resistance, though it can also occur naturally through genetic mutations and the spread of resistant genes. Microbes resistant to multiple drugs are termed multidrug-resistant (MDR) and are sometimes called superbugs. 

[Traditional ecological knowledge] Traditional ecological knowledge (TEK) is a cumulative body of knowledge, practice, and belief, evolving by adaptive processes and handed down through generations by cultural transmission, about the relationship of living beings with one another and with their environment. 

  

[Brown v. Board of Education] Brown v. Board of Education of Topeka, 347 U.S. 483 (1954), is a landmark decision of the United States Supreme Court that ruled that U.S. state laws establishing racial segregation in public schools violate the Equal Protection Clause of the Fourteenth Amendment, even if the segregated facilities are equal in quality. The decision partially overruled the Court's 1896 decision Plessy v. Ferguson, which had ruled that racial segregation laws were constitutional as long as the facilities for each race were equal, a doctrine that had come to be known as "separate but equal". The Court's unanimous decision in Brown and its related cases paved the way for integration and was a major victory of the civil rights movement, and it became a model for many future impact litigation cases. 

  

[Photolithography] Photolithography is a process that involves using light to transfer a pattern onto a photoresist layer deposited on a sample, typically a silicon wafer. It is used in the manufacturing of integrated circuits. 

  

[Ubuntu philosophy] Ubuntu describes a set of closely related Bantu African-origin value systems that emphasize the interconnectedness of individuals with their surrounding societal and physical worlds. "Ubuntu" is sometimes translated as "I am because we are". In Xhosa, the latter term is used, but is often meant in a more philosophical sense to mean "the belief in a universal bond of sharing that connects all humanity". 

  

[Go (game)] Go, Weiqi, or Baduk is an abstract strategy board game for two players in which the aim is to fence off more territory than the opponent. The game was invented in China more than 2,500 years ago and is believed to be the oldest board game continuously played to the present day. A 2016 survey by the International Go Federation's 75 member nations found that there are over 46 million people worldwide who know how to play Go, and over 20 million current players, the majority of whom live in East Asia. 

[Ancient Egypt] Ancient Egypt was a cradle of civilization concentrated along the lower reaches of the Nile River in the eastern part of North Africa. It emerged from prehistoric Egypt around 3150 BC, when Upper and Lower Egypt were united by Menes, who is believed by the majority of Egyptologists to have been the same person as Narmer. The history of ancient Egypt unfolded as a series of stable kingdoms interspersed by the "Intermediate Periods" of relative instability. These stable kingdoms existed in one of three periods: the Old Kingdom of the Early Bronze Age; the Middle Kingdom of the Middle Bronze Age; or the New Kingdom of the Late Bronze Age. 

  

[Ancient Greece] Ancient Greece was an ancient civilization existing from the Greek Dark Ages of the 12th–9th centuries BC to the end of classical antiquity, comprising a loose collection of culturally and linguistically related city-states and regions, primarily centered in the northeastern area of the Mediterranean Sea. Prior to the Roman period, most of this area was officially unified only once, from 338 to 323 BC under the Kingdom of Macedon. In Western history, the era of classical antiquity was immediately followed by the Early Middle Ages and the Byzantine period. 

  

[Ancient Rome] In modern historiography, ancient Rome is the Roman civilisation from the founding of the Italian city of Rome in the 8th century BC to the collapse of the Western Roman Empire in the 5th century AD. It encompasses the Roman Kingdom (753–509 BC), the Roman Republic (509‍–‍27 BC), and the Roman Empire until the fall of the western empire. 

  

[Roman Empire] The Roman Empire was a state that dominated the Mediterranean and much of Europe, Western Asia, and North Africa during the classical period. The Roman Republic had previously conquered most of these territories, which later came under permanent single-person rule following Octavian's rise to power and the establishment of the Augustan Principate in 27 BC. By the late 3rd century AD, the empire had been divided many times. The Western Roman Empire collapsed in 476 AD, whereas the Eastern Roman Empire persisted until the fall of Constantinople in 1453. 

  

[Byzantine Empire] The Byzantine Empire, also known as the Eastern Roman Empire, was the continuation of the Roman Empire centred on Constantinople during late antiquity and the Middle Ages. Having survived the fall of the Western Roman Empire in the 5th century AD, it endured until the fall of Constantinople to the Ottoman Empire in 1453. The term 'Byzantine Empire' was coined only after its demise; its citizens used the term 'Roman Empire' and called themselves 'Romans'. 

  

[Maya civilization] The Maya civilization was a Mesoamerican civilization that existed from antiquity to the early modern period. Known by its ancient temples and glyphs (script), the civilization is also noted for its art, architecture, mathematics, calendar, and astronomical system. The Maya script is the most sophisticated and highly developed writing system in the pre-Columbian Americas. 

  

[Aztecs] The Aztecs were a Mesoamerican civilization that flourished in central Mexico from about 1300 to 1521. The Aztec people included different ethnic groups of central Mexico, particularly those who spoke Nahuatl. Aztec culture was organized into city-states (altepetl), some of which joined to form alliances, political confederations, or empires. The Aztec Empire was a confederation of three city-states established in 1427: Tenochtitlan, Tetzcoco, and Tlacopan. Although the term Aztecs is often narrowly restricted to the Mexica of Tenochtitlan, it is also broadly used to refer to Nahua polities or peoples of central Mexico in the pre-Hispanic era, as well as the Spanish colonial era (1521–1821). 

  

[Inca Empire] The Inca Empire, officially known as the Realm of the Four Parts, was the largest empire in pre-Columbian America. The administrative, political, and military center of the empire was in the city of Cusco. The Inca civilisation rose from the Peruvian highlands sometime in the early 13th century. The Portuguese explorer Aleixo Garcia was the first European to reach the Inca Empire in 1524. Later, in 1532, the Spanish began the conquest of the Inca Empire, and by 1572 the last Inca state was fully conquered. 

  

[Han dynasty] The Han dynasty was an imperial dynasty of China established by Liu Bang, and preceded by the short-lived Qin dynasty (221–206 BC) and the interregnum known as the Chu–Han Contention (206–202 BC). It was succeeded by the Three Kingdoms period (220–280 AD) and also briefly interrupted by the Xin dynasty (9–23 AD) established by the usurping regent Wang Mang. It is thus separated into two periods—the Western Han and the Eastern Han (25–220 AD). The Han dynasty is considered a golden age in Chinese history, impacting Chinese identity in later periods. The majority ethnic group of modern China refer to themselves as the "Han people", while spoken Chinese and written Chinese are referred to respectively as the "Han language" and "Han characters". 

  

[Abbasid Caliphate] The Abbasid Caliphate or Abbasid Empire was the third Islamic caliphate, ruled by the Abbasid dynasty. The dynasty was descended from Muhammad's uncle, Abbas ibn Abd al-Muttalib, after whom it is named. The Abbasids rose to power in 750, when the Abbasid Revolution overthrew the preceding Umayyad Caliphate, and they ruled as caliphs from their base in Iraq until 1258, with Baghdad as their capital for most of their history. 

  

[Mongol Empire] The Mongol Empire was the largest contiguous empire in history. Originating in present-day Mongolia in East Asia, the medieval empire at its height stretched from the Sea of Japan to Eastern Europe, extending northward into Siberia and east and southward into the Indian subcontinent, mounting invasions of Southeast Asia, and conquering the Iranian Plateau; and reaching westward as far as the Levant and the Carpathian Mountains. 

  

[Holy Roman Empire] The Holy Roman Empire, also known as the Holy Roman Empire of the German Nation after 1512, was a polity comprising and controlling much of Central Europe and Western Europe, headed by the Holy Roman Emperor and characterized by a decentralized political structure. It developed in the Early Middle Ages, and lasted for a millennium until its dissolution in 1806 during the Napoleonic Wars. Initially, it consisted of three parts—Germany, Italy, and Burgundy—held together by the emperor's overlordship. By the 15th century, imperial governance had become concentrated in and upon the Kingdom of Germany, as the empire's effective control over Italy and Burgundy had largely disappeared. 

  

[Qing dynasty] The Qing dynasty, officially the Great Qing, also known as the Qing Empire or Qing China, was a Manchu-led imperial dynasty of China and an early modern empire in East Asia which existed from 1636/1644 to 1912. The last imperial dynasty in Chinese history, the Qing dynasty was preceded by the Ming dynasty and succeeded by the Republic of China. At the height of its power, the empire stretched from the Sea of Japan in the east to the Pamir Mountains in the west, and from the Mongolian Plateau in the north to the South China Sea in the south. Originally emerging from the Later Jin dynasty founded in 1616 and proclaimed in Shenyang in 1636, the dynasty seized control of the Ming capital Beijing and North China in 1644, traditionally considered the start of the dynasty's rule. The dynasty lasted until the Xinhai Revolution of October 1911 led to the abdication of the last emperor in February 1912. The multi-ethnic Qing dynasty assembled the territorial base for modern China. The Qing controlled the most territory of any dynasty in Chinese history, and in 1790 was the fourth-largest empire in world history to that point. It was also the most populous state at the time, with over 426 million citizens in 1907. 

  

[Achaemenid Empire] The Achaemenid Empire was an ancient Iranian empire founded by Cyrus the Great of the Achaemenid dynasty in 550 BC. At peak, its territorial extent was roughly 5.5 million square kilometres, making it the largest empire of its time. Based in the Iranian plateau, it stretched from the Balkans and Cyrenaica in the west to the Indus Valley in the east, including Anatolia, Cyprus, Mesopotamia, the Levant, the South Caucasus, parts of Eastern Arabia, and large parts of Central Asia. 

  

[Silk Road] The Great Silk Road was a network of Asian trade routes active from the second century BCE until the mid-15th century. Spanning over 6,400 km (4,000 mi) on land, it played a central role in facilitating economic, cultural, political, and religious interactions between the Eastern and Western worlds. The name "Silk Road" was coined in the late 19th century, but some 20th- and 21st-century historians instead prefer the term Silk Routes, on the grounds that it more accurately describes the intricate web of land and sea routes connecting Central, East, South, Southeast, and West Asia as well as East Africa and Southern Europe. In fact, some scholars criticise or even dismiss the idea of silk roads and call for a new definition or alternate term. According to them, the literature using this term has "privileged the sedentary and literate empires at either end of Eurasia" thereby ignoring the contributions of steppe nomads. In addition, the classic definition sidelines prominent civilisations such as India and Iran. 

  

[Middle Ages] In the history of Europe, the Middle Ages or medieval period lasted approximately from the 5th to late 15th centuries, comparable with the post-classical period of global history. The medieval period is the middle epoch of the three traditional divisions of Western history: classical antiquity, the medieval period, and the modern period. The medieval period is itself subdivided into the Early, High, and Late Middle Ages. 

  

[Neolithic Revolution] The Neolithic Revolution, also known as the  First Agricultural Revolution, was the wide-scale transition of many human cultures during the Neolithic period in Afro-Eurasia from a lifestyle of hunting and gathering to one of agriculture and settlement, making an increasingly large population possible. These settled communities permitted humans to observe and experiment with plants, learning how they grew and developed. This new knowledge led to the domestication of plants into crops. 

  

[French Revolution] The French Revolution was a period of political and societal change in France that began with the Estates General of 1789 and ended with the Coup of 18 Brumaire on 9 November 1799. Many of the revolution's ideas are considered fundamental principles of liberal democracy, and its values remain central to modern French political discourse. It was caused by a combination of social, political, and economic factors which the existing regime proved unable to manage. 

  

[Industrial Revolution] The Industrial Revolution, sometimes called the First Industrial Revolution in contrast to the subsequent Second Industrial Revolution, was a transitional period of the global economy toward more widespread, efficient and stable manufacturing processes, succeeding the Second Agricultural Revolution. Beginning in Great Britain around 1760, the Industrial Revolution had spread to continental Europe and the United States by about 1840. Economic historians agree that the onset of the Industrial Revolution is the most important event in human history, comparable only to the adoption of agriculture with respect to material advancement. 

  

[World War I] World War I, or the First World War, also known as the Great War, was a global conflict between two coalitions: the Allies and the Central Powers. One of the deadliest conflicts in history, World War I resulted in an estimated 15 to 22 million deaths, including those in war crimes and genocides. The war also helped spread the Spanish flu pandemic. The conflict saw important developments in weaponry, including the first large-scale use of machine guns, artillery, aircraft, chemical weapons, and tanks. 

  

[World War II] World War II, or the Second World War, was a global conflict between two coalitions: the Allies and the Axis powers. Nearly all of the world's countries participated, with many engaging in total war on an unprecedented scale. World War II was the deadliest conflict in history, causing the deaths of 60 to 75 million people, a majority of whom were civilians. Millions died as a result of massacres, starvation, disease, and genocides including the Holocaust. After the Allied victory, Germany, Austria, Japan, and Korea were occupied, and German and Japanese leaders were tried for war crimes. 

  

[Cold War] The Cold War was a period of international geopolitical rivalry between the United States (US) and the Soviet Union (USSR) and their respective allies, the capitalist Western Bloc and communist Eastern Bloc. It began in the aftermath of the Second World War and ended with the dissolution of the Soviet Union in 1991. The term cold war is used because there was no direct fighting between the two superpowers, though each supported opposing sides in regional conflicts known as proxy wars. In addition to the struggle for ideological and economic influence and an arms race in both conventional and nuclear weapons, the Cold War was expressed through technological rivalries such as the Space Race, espionage, propaganda campaigns, embargoes, and sports diplomacy. 

  

[Christianity]  

Christianity is an Abrahamic monotheistic religion based on the Bible and the teachings of Jesus. The Gospels state that Jesus is the Son of God and rose from the dead after his crucifixion, whose coming as the messiah (Christ) was prophesied in the Old Testament and chronicled in the New Testament. It is the world's largest and most widespread religion with over 2.3 billion followers, comprising around 28.8% of the world population. Its adherents, known as Christians, are estimated to make up a majority of the population in 120 countries and territories. 

  

[Islam] Islam is an Abrahamic religion based on the Quran and the teachings of Muhammad. The monotheistic religion has an estimated 2 billion worldwide adherents, called Muslims. Islam is the world's second-largest religious population after Christianity. 

  

[Hinduism] Hinduism is an umbrella term for a range of Indian religious and spiritual traditions (sampradayas) that are unified by their adherence to Dharma, a cosmic order maintained by its followers through rituals and righteous living, as expounded in the Vedas. The word Hindu originated as an exonym, and while Hinduism has been called the oldest surviving religion in the world, it is also described since the 19th century by the term Sanātana Dharma. Vaidika Dharma and Arya Dharma are historical endonyms for Hinduism. 

  

[Judaism] Judaism is an Abrahamic, monotheistic, ethnic religion that comprises the collective spiritual, cultural, and legal traditions of the Jewish people. Religious Jews regard Judaism as their means of observing the Mosaic covenant, which they believe was established between God and the Jewish people. The religion is considered one of the earliest monotheistic religions. 

  

[Sikhism] Sikhism, also known as Sikhi, is an Indian religion and philosophy that originated in the Punjab region of the Indian subcontinent around the end of the 15th century CE. It is one of the most recently founded major religions and is followed by 25–30 million adherents, known as Sikhs. 

  

[Taoism] Taoism or Daoism is a philosophical and religious tradition indigenous to China, emphasizing harmony with the Tao 道. With a range of meanings and interpretations in Chinese philosophy, translations of Tao include 'way', 'road', 'path', or 'technique', generally understood in the Taoist sense as an enigmatic process of transformation ultimately underlying reality. Taoist thought has informed the development of various practices within the Taoist tradition, including forms of meditation, astrology, qigong, feng shui, and internal alchemy. A common goal of Taoist practice is self-cultivation, a deeper appreciation of the Tao, and more harmonious existence. Taoist ethics generally emphasize virtues of effortless action, naturalness, simplicity, and the three treasures of compassion, frugality, and humility. 

  

[Confucianism] Confucianism, also known as Ruism or Ru classicism, is a system of thought and behavior originating in ancient China, and is variously described as a tradition, philosophy, religion, theory of government, or way of life. Founded by Confucius in the Hundred Schools of Thought era, Confucianism integrates philosophy, ethics, and social governance, with a core focus on virtue, social harmony, and familial responsibility. 

  

[Shinto] Shinto , also called Shintoism, is a religion from Japan. Classified as an East Asian religion by scholars of religion, it is often regarded by its practitioners as Japan's indigenous religion and as a nature religion. Scholars sometimes call its practitioners Shintoists, although adherents rarely use that term themselves. With no central authority in control of Shinto, there is much diversity of belief and practice evident among practitioners. 

  

[Zoroastrianism] Zoroastrianism, also called Mazdayasna and Behdin, is an Iranian religion centred on the Avesta and the teachings of Zarathushtra Spitama, who is more commonly referred to by the Greek translation, Zoroaster. Among the world's oldest organized faiths, its adherents exalt an uncreated, benevolent, and all-wise deity known as Ahura Mazda, who is hailed as the supreme being of the universe. 

  

[Jainism]  

Jainism, also known as Jain Dharma, is an Indian religion that teaches a path toward spiritual purity and enlightenment through disciplined nonviolence to all living creatures. The tradition is spiritually guided by 24 tirthankaras (ford-makers), supreme teachers who have conquered the cycle of rebirth and attained omniscience. The core of Jain philosophy is established on three ethical pillars: ahiṃsā (nonviolence), anekāntavāda, and aparigraha (non-possession). While its ultimate spiritual goal is moksha, these ethical principles have historically fostered a community renowned for its high literacy, trusted role in commerce, and distinct intellectual culture. 

  

[Stoicism] Stoicism is a philosophical movement and practical guide to living, emphasizing daily self-discipline and moral improvement, which originated in the Hellenistic period of ancient Greece and proliferated well into the Roman Imperial period. The ancient Stoics believed that the universe operated according to reason, or logos, providing a unified account of the world, constructed from ideals of rational discourse, monistic physics, and naturalistic ethics. These ideals constitute virtue, which is necessary for the Stoic goal of 'living a well-reasoned life'. 

  

[Existentialism] Existentialism is a family of philosophical views and inquiry that explore the human individual's struggle to lead an authentic life despite the apparent absurdity or incomprehensibility of existence. In examining meaning, purpose, and value, existentialist thought often includes concepts such as existential crises, angst, courage, and freedom. 

  

[Utilitarianism] Utilitarianism is a family of theories in normative ethics that prescribe actions that maximize happiness and well-being for the affected individuals. In other words, utilitarian ideas encourage actions that lead to the greatest good for the greatest number. Although different varieties of utilitarianism admit different characterizations, the basic idea that underpins them all is, in some sense, to maximize utility, which is often defined in terms of well-being or related concepts. For instance, Jeremy Bentham, the founder of utilitarianism, described utility as the capacity of actions or objects to produce benefits, such as pleasure, happiness, and good, or to prevent harm, such as pain and unhappiness, to those affected. 

  

[Epistemology]  

  

Epistemology is the branch of philosophy that examines the nature, origin, and limits of knowledge. Also called the theory of knowledge, it explores different types of knowledge, such as propositional knowledge about facts, practical knowledge in the form of skills, and knowledge by acquaintance as a familiarity through experience. Epistemologists study the concepts of belief, truth, and justification to understand the nature of knowledge. To discover how knowledge arises, they investigate sources of justification, such as perception, introspection, memory, reason, and testimony. 

  

[Metaphysics] Metaphysics is the branch of philosophy that examines the basic nature or most fundamental structure of reality. It is traditionally seen as the study of mind-independent features of the world, but some theorists view it as an inquiry into the conceptual framework of human understanding. Some philosophers, including Aristotle, designate metaphysics as the first philosophy to suggest that it is more fundamental than other forms of philosophical inquiry. 

  

[Free will] Free will is generally understood as the capacity or ability of people to (a) choose between different possible courses of action, (b) exercise control over their actions in a way that is necessary for moral responsibility, or (c) be the ultimate source or originator of their actions. There are different theories as to its nature, and these aspects are often emphasized differently depending on philosophical tradition, with debates focusing on whether and how such freedom can coexist with physical determinism, divine foreknowledge, and other constraints. 

  

[Social contract] In moral and political philosophy, the social contract is an idea, theory, or model that usually, although not always, concerns the legitimacy of the authority of the state over the individual. Conceptualized in the Age of Enlightenment, it is a core concept of constitutionalism, while not necessarily convened and written down in a constituent assembly and constitution. 

  

[Nihilism] Nihilism is a family of philosophical views that reject the existence of any objectively meaningful purpose, moral value, truth, knowledge, or related concepts. Nihilistic views span several branches of philosophy, including ethics, value theory, epistemology, and metaphysics. Nihilism is also described as a broad cultural phenomenon or historical movement that pervades modernity in the Western world. 

  

[Impressionism] Impressionism was a 19th-century art movement characterised by visible brush strokes, open composition, emphasis on accurate depiction of light in its changing qualities, ordinary subject matter, unusual visual angles, and inclusion of movement as a crucial element of human perception and experience. Impressionism originated with a group of Paris-based artists whose independent exhibitions brought them to prominence during the 1870s and 1880s. 

  

[Cubism] Cubism is an early-20th-century avant-garde art movement which began in Paris. It revolutionized painting and the visual arts, and sparked artistic innovations in music, ballet, literature, and architecture. 

  

[Surrealism] Surrealism is an art and cultural movement that developed in Europe in the aftermath of World War I in which artists aimed to allow the unconscious mind to express itself, often resulting in the depiction of illogical or dreamlike scenes and ideas. Its intention was, according to leader André Breton, to "resolve the previously contradictory conditions of dream and reality into an absolute reality, a super-reality", or surreality. It produced works of painting, writing, photography, theatre, filmmaking, music, comedy and other media as well. 

  

[Baroque] The Baroque is a Western style of architecture, music, dance, painting, sculpture, poetry, and other arts that flourished from the early 1600s until the 1750s. It followed Renaissance art and Mannerism and preceded the Rococo and Neoclassical styles. It was encouraged by the Catholic Church as a means to counter the simplicity and austerity of Protestant architecture, art, and music, though Lutheran Baroque art developed in parts of Europe as well. 

  

[Gothic architecture] Gothic architecture is an architectural style prevalent in Europe from the late 12th to the 16th century, during the High and Late Middle Ages, surviving into the 17th and 18th centuries in some areas. It evolved from Romanesque architecture and was succeeded by Renaissance architecture. It originated in the Île-de-France and Picardy regions of northern France. The style at the time was sometimes known as opus Francigenum ; the term Gothic was first applied contemptuously during the later Renaissance, by those ambitious to revive the architecture of classical antiquity. The defining design element of Gothic architecture is the pointed arch. The use of the pointed arch in turn led to the development of the pointed rib vault and flying buttresses, combined with elaborate tracery and stained glass windows. 

  

[Pop art] Pop art is an art movement that emerged in the United Kingdom and the United States during the mid-to-late 1950s. The movement presented a challenge to traditions of fine art by including imagery from popular and mass culture—including advertising, comic strips, product packaging, celebrities, and everyday consumer goods—into painting, sculpture, and printmaking. By elevating the banal, the kitsch, and the mass-produced to the status of high art, pop art blurred the boundaries between high and low culture. It is also associated with the artists' use of mechanical means of reproduction or rendering techniques. In pop art, material is sometimes visually removed from its known context, isolated, or combined with unrelated material. 

  

[Art Nouveau] Art Nouveau is an international style of art, architecture, and applied art, especially the decorative arts. It was often inspired by natural forms such as the sinuous curves of plants and flowers. Other characteristics of Art Nouveau were a sense of dynamism and movement, often given by asymmetry or whiplash lines, and the use of modern materials, particularly iron, glass, ceramics and later concrete, to create unusual forms and larger open spaces. It was popular between 1890 and 1910 during the Belle Époque period, and was a reaction against the academicism, eclecticism and historicism of 19th-century architecture and decorative art. 

  

[Bauhaus] The Staatliches Bauhaus, commonly known as the Bauhaus, was a German art school operational from 1919 to 1933 that combined crafts and the fine arts. The school became famous for its approach to design, which attempted to unify individual artistic vision with the principles of mass production and emphasis on function. 

  

[Leonardo da Vinci] Leonardo di ser Piero da Vinci was an Italian polymath of the High Renaissance who was active as a painter, draughtsman, engineer, scientist, theorist, sculptor, and architect. While his fame initially rested on his achievements as a painter, he has also become known for his notebooks, in which he made drawings and notes on a variety of subjects, including anatomy, astronomy, botany, cartography, painting, and palaeontology. Leonardo is widely regarded as a genius who epitomised the Renaissance humanist ideal, and his collective works contributed to the development of European art to an extent rivalled only by that of his younger contemporary Michelangelo. 

  

[William Shakespeare] William Shakespeare was an English playwright, poet and actor. He is widely regarded as the greatest writer in the English language and the world's pre-eminent dramatist. He is often called England's national poet and the "Bard of Avon" or simply "the Bard". His extant works, including collaborations, consist of some 39 plays, 154 sonnets, three long narrative poems and a few other verses, some of uncertain authorship. His plays have been translated into every major living language and are performed more often than those of any other playwright. Shakespeare remains arguably the most influential writer in the English language, and his works continue to be studied and reinterpreted. 

  

[Ludwig van Beethoven] Ludwig van Beethoven was a German composer, conductor, and pianist. Regarded as one of the greatest composers in the history of Western music, he was mentored during the Classical period, and his musical style was a key driver of the transition to Romantic music, and the expansion of instrumental forms such as the symphony, the piano sonata and the string quartet. His compositions have attracted extraordinary casual and scholarly interest, and remain among the most performed in the world. 

  

[Wolfgang Amadeus Mozart] Wolfgang Amadeus Mozart was a Classical composer and musician. He completed more than 800 works in his life—including outstanding examples of most of the genres of his time: symphonies, concertos, chamber music, opera and choral music—and is regarded as one of the greatest composers in the history of Western music. 

  

[Isaac Newton] Sir Isaac Newton was an English polymath who was a mathematician, physicist, astronomer, alchemist, theologian, author and inventor. He was a key figure in the Scientific Revolution and the Enlightenment that followed. His book Philosophiæ Naturalis Principia Mathematica, first published in 1687, achieved the first great unification in physics and established classical mechanics. Newton also made seminal contributions to optics, and shares credit with the German mathematician Gottfried Wilhelm Leibniz for formulating infinitesimal calculus, although he developed calculus years before Leibniz. Newton contributed to and refined the scientific method, and his work is considered the most influential in bringing forth modern science. 

  

[Albert Einstein] Albert Einstein was a German-born theoretical physicist best known for developing the theory of relativity. Einstein also made important contributions to quantum theory. His mass–energy equivalence formula E = mc2, which arises from special relativity, has been called "the world's most famous equation". He received the 1921 Nobel Prize in Physics for "his services to theoretical physics, and especially for his discovery of the law of the photoelectric effect". 

  

[Charles Darwin] Charles Robert Darwin was an English naturalist, geologist, and biologist, widely known for his contributions to evolutionary biology. His proposition that all species of life have descended from a common ancestor is now generally accepted and considered a fundamental scientific concept. In a joint presentation with Alfred Russel Wallace, he introduced his scientific theory that this branching pattern of evolution resulted from a process he called natural selection, in which the struggle for existence has a similar effect to the artificial selection involved in selective breeding. Darwin has been described as one of the most influential figures in human history and was honoured by burial in Westminster Abbey. 

  

[Marie Curie] Maria Salomea Skłodowska Curie, better known as Marie Curie was a Polish and naturalised-French physicist and chemist. She shared the 1903 Nobel Prize in Physics with her husband, Pierre Curie, "for their joint researches on the radioactivity phenomena discovered by Professor Henri Becquerel". She won the 1911 Nobel Prize in Chemistry "[for] the discovery of the elements radium and polonium, by the isolation of radium and the study of the nature and compounds of this remarkable element". 

  

[Nikola Tesla] Nikola Tesla was a Serbian-American engineer, futurist, and inventor. He is known for his contributions to the design of the modern alternating current (AC) electricity supply system. 

  

[Ada Lovelace] Augusta Ada King, Countess of Lovelace, also known as Ada Lovelace, was an English mathematician and writer chiefly known for work on Charles Babbage's proposed mechanical general-purpose computer, the analytical engine. She was the first to recognise the machine had applications beyond pure calculation. Lovelace is often considered the first computer programmer. 

  

[Alan Turing] Alan Mathison Turing was an English mathematician, computer scientist, logician, cryptanalyst, philosopher and theoretical biologist. He was highly influential in the development of theoretical computer science, providing a formalisation of the concepts of algorithm and computation with the Turing machine, which can be considered a model of a general-purpose computer. Turing is widely considered to be the father of theoretical computer science. 

  

[Alexander the Great] Alexander III of Macedon, most commonly known as Alexander the Great, was king of the ancient Greek kingdom of Macedon from 336 BC until his death. He was the son of Philip II, born to him by his fourth wife, Olympias, and ascended to the throne at the age of 20 after Philip was assassinated. Alexander spent most of his reign conducting a lengthy military campaign throughout Asia and Egypt, and by the age of 30, he had created one of the largest empires in history, stretching from Greece to northwestern India. He was undefeated in battle and is widely considered to be one of history's greatest and most successful military commanders. 

  

[Julius Caesar] Gaius Julius Caesar was a Roman general, statesman, and author who was the dictator of the Roman Republic almost continuously from 49 BC until his assassination in 44 BC. A member of the First Triumvirate, he led the Roman armies through the Gallic Wars and defeated his political rival Pompey in Caesar's civil war. He consolidated power and was titled as dictator perpetuo in 44 BC, helping create the political conditions that led to the collapse of the Roman Republic and the emergence of the Roman Empire. For his role in these events, he is regarded as one of history's most influential figures. 

  

[Cleopatra] Cleopatra VII Thea Philopator was Queen of the Ptolemaic Kingdom of Egypt from 51 to 30 BC, and the last active Hellenistic pharaoh. A member of the Ptolemaic dynasty, she was a descendant of its founder Ptolemy I Soter, a Macedonian Greek general and companion of Alexander the Great. Her first language was Koine Greek, and she is the only Ptolemaic ruler known to have learned the Egyptian language, among several others. After her death, Egypt became a province of the Roman Empire, marking the end of the Hellenistic period in the Mediterranean, which had begun during the reign of Alexander. 

  

[Genghis Khan] Genghis Khan, also known as Chinggis Khan, was the founder and first khan of the Mongol Empire. After spending most of his life uniting the Mongol tribes, he launched a series of military campaigns, conquering large parts of China and Central Asia. 

  

[Napoleon] Napoleon Bonaparte, later known by his regnal name Napoleon I, was Emperor of the French from 18 May 1804 until his first abdication in 1814, with a brief restoration during the Hundred Days in 1815. He rose to prominence as a general during the French Revolution and led a series of military campaigns across Europe and the Middle East during the French Revolutionary and Napoleonic Wars. As a statesman, he implemented numerous legal and administrative reforms in France and Europe. 

  

[Mahatma Gandhi] Mohandas Karamchand Gandhi was an Indian lawyer, anti-colonial nationalist and political ethicist who employed nonviolent resistance to lead the successful campaign for India's independence from British rule, and to later inspire movements for civil rights and freedom across the world. The honorific Mahātmā, first applied to him in 1914 in South Africa, is now used throughout the world. 

  

[Nelson Mandela] Nelson Rolihlahla Mandela was a South African anti-apartheid revolutionary, politician, and philanthropist, who served as President of South Africa from 1994 to 1999. He was the country's first black head of state and the first elected in a fully representative democratic election. His government focused on dismantling the legacy of apartheid by tackling institutionalised racism and fostering racial reconciliation. Ideologically an African nationalist and socialist, he served as President of the African National Congress (ANC) party from 1991 to 1997. 

  

[Martin Luther King Jr.] Martin Luther King Jr. was an American civil rights activist and Baptist minister who was a prominent leader of the civil rights movement from 1955 until his assassination in 1968. He advanced civil rights for people of color in the United States through the use of nonviolent resistance and civil disobedience against Jim Crow laws and other forms of legalized discrimination, which most commonly affected African Americans. 

  

[Abraham Lincoln] Abraham Lincoln was the 16th president of the United States, serving from 1861 until his assassination in 1865. He led the United States through the American Civil War, defeating the Confederacy and playing a major role in the abolition of slavery. 

  

[Pablo Picasso] Pablo Ruiz Picasso was a Spanish painter and sculptor who spent most of his adult life in France. One of the most influential artists of the 20th century, he is known for co-founding the Cubist movement, the invention of constructed sculpture, the co-invention of collage, and for the wide variety of styles that he helped develop and explore. Among his most famous works are the proto-Cubist Les Demoiselles d'Avignon (1907) and the anti-war painting Guernica (1937), a dramatic portrayal of the bombing of Guernica by German and Italian air forces during the Spanish Civil War. His career spanned more than 76 years, from his late teens to his death in 1973. 

  

[Jane Austen] Jane Austen was an English writer known primarily for her six novels, which implicitly interpret, critique, and comment on the English landed gentry at the end of the 18th century. 

  

[Leo Tolstoy] Count Lev Nikolayevich Tolstoy, usually referred to in English as Leo Tolstoy, was a Russian writer. He is regarded as one of the greatest and most influential authors of all time. 

  

[Fyodor Dostoevsky] Fyodor Mikhailovich Dostoevsky was a Russian philosopher, novelist, short-story writer, essayist and journalist. He is regarded as one of the greatest novelists in both Russian and world literature, and many of his works are considered highly influential masterpieces. Dostoevsky's literary works explore the human condition in the troubled political, social and spiritual atmospheres of 19th-century Russia, and engage with a variety of philosophical and religious themes. His most acclaimed novels include Crime and Punishment (1866), The Idiot (1869), Demons (1872), The Adolescent (1875) and The Brothers Karamazov (1880). His Notes from Underground, a novella published in 1864, is considered one of the first works of existentialist literature. 

  

[Poetry] Poetry is a form of literary art that uses aesthetic and often rhythmic qualities of language to evoke meanings in addition to, or in place of, literal or surface-level meanings. A particular instance of poetry is a poem and is written by a poet. Poets use a variety of techniques called poetic devices, such as assonance, alliteration, consonance, euphony and cacophony, onomatopoeia, rhythm, rhyme schemes and sound symbolism, to produce musical or other artistic effects. They also frequently organize these devices into poetic structures, which may be strict or loose, conventional or invented by the poet. Poetic structures vary dramatically by language and cultural convention, but they often rely on rhythmic metre: patterns of syllable stress or syllable weight. They may also use repeating patterns of phonemes, phoneme groups, tones, words, or entire phrases. Poetic structures may even be semantic. 

  

[Novel] A novel is an extended work of narrative fiction usually written in prose and published as a book. The word derives from the Italian: novella for 'new', 'news', or 'short story ', itself from the Latin: novella, a singular noun use of the neuter plural of novellus, diminutive of novus, meaning 'new'. According to Margaret Doody, the novel has "a continuous and comprehensive history of about two thousand years", with its origins in the Ancient Greek and Roman novel, Medieval chivalric romance, and the tradition of the Italian Renaissance novella. The ancient romance form was revived by Romanticism, in the historical romances of Walter Scott and the Gothic novel. Some novelists, including Nathaniel Hawthorne, Herman Melville, Ann Radcliffe, and John Cowper Powys, preferred the term romance. Such romances should not be confused with the genre fiction romance novel, which focuses on romantic love. M. H. Abrams and Walter Scott have argued that a novel is a fiction narrative that displays a realistic depiction of the state of a society, like Harper Lee's To Kill a Mockingbird. The romance, on the other hand, encompasses any fictitious narrative that emphasizes marvellous or uncommon incidents. In reality, such works are nevertheless also commonly called novels, including Mary Shelley's Frankenstein and J. R. R. Tolkien's The Lord of the Rings. 

  

[Science fiction] Science fiction is the genre of speculative fiction that imagines advanced and futuristic changes in technology, scientific knowledge, or biological systems. The elements common to science fiction have increased over time: from space exploration, extraterrestrial life, time travel, and robotics; to parallel universes, dystopian societies, and biological manipulations; and, most recently, to information technology, transhumanism, posthumanism, and environmental challenges. Science fiction often explores human responses to the consequences of these types of projected or imagined scientific advances. 

  

[Myth] Myth is a genre of folklore consisting primarily of narratives that play a fundamental role in a society. For scholars, this is totally different from the ordinary sense of the term myth, meaning a belief that is not true, as the veracity of a piece of folklore is entirely irrelevant to determining whether it constitutes a myth. 

  

[Hamlet] The Tragedy of Hamlet, Prince of Denmark, often shortened to Hamlet, is a tragedy written by William Shakespeare sometime between 1599 and 1601. It is Shakespeare's longest play. Set in Denmark, the play depicts Prince Hamlet and his attempts to exact revenge against his uncle, Claudius, who has murdered Hamlet's father in order to seize his throne and marry Hamlet's mother. 

  

[Divine Comedy] The Divine Comedy is an Italian narrative poem by Dante Alighieri, begun c. 1308 and completed c. 1321, shortly before the author's death. It is widely considered the pre-eminent work in Italian literature and one of the greatest works of Western literature. The poem's imaginative vision of the afterlife is representative of the medieval worldview as it existed in the Western Church by the 14th century. It helped establish the Tuscan language, in which it is written, as the standardized Italian language. It is divided into three parts: Inferno, Purgatorio, and Paradiso. 

  

[Iliad] The Iliad is one of two major surviving ancient Greek epic poems attributed to Homer. It is one of the oldest extant works of literature still widely read by modern readers. Like the Odyssey, the poem is divided into 24 books and was written in dactylic hexameter. It contains 15,693 lines in its standard edition. The Iliad is often regarded as the first substantial piece of European literature and is central to the study of classical philology. 

  

[Odyssey] The Odyssey is one of two major epics of ancient Greek literature attributed to Homer. It is one of the oldest surviving works of literature and remains popular with modern audiences. Like the Iliad, the Odyssey is divided into 24 books. It follows the heroic king of Ithaca, Odysseus, also known by the Latin variant Ulysses, and his homecoming journey after the ten-year-long Trojan War. His journey from Troy to Ithaca lasts an additional ten years, during which time he encounters many perils and all of his crewmates are killed. In Odysseus's long absence, he is presumed dead, leaving his wife Penelope and son Telemachus to contend with a group of unruly suitors competing for Penelope's hand in marriage. 

  

[Classical music] Classical music is a tradition of art music in the Western world, considered to be distinct from Western folk music or popular music. It is sometimes distinguished as Western classical music, as the term "classical music" can also be applied to non-Western art musics. Classical music is often characterized by formality and complexity in its musical form and harmonic organization, particularly with the use of polyphony. Since at least the ninth century, it has been primarily a written tradition, spawning a sophisticated notational system, as well as accompanying literature in analytical, critical, historiographical, musicological and philosophical practices. 

  

[Opera] Opera is a form of Western theatre in which music is a fundamental component and dramatic roles are taken by singers. Such a "work" is typically a collaboration between a composer and a librettist and incorporates a number of the performing arts, such as acting, scenery, costume, and sometimes dance or ballet. The performance is typically given in an opera house, accompanied by an orchestra or smaller musical ensemble, which since the early 19th century has been led by a conductor. Although musical theatre is closely related to opera, the two are considered to be distinct from one another. 

  

[Rock music] Rock music is a genre of popular music that originated in the United States as "rock and roll" in the late 1940s and early 1950s, developing into a range of styles from the late 50s to mid-1960s, primarily in the United States and United Kingdom. It has its roots beginning as classic rock and roll, a style that drew from the African-American musical genres of blues and rhythm and blues, as well as from country music. Rock also drew strongly from genres such as electric blues and folk, and incorporated influences from jazz and other styles. Rock is typically centered on the electric guitar, usually as part of a rock group with an electric bass guitar, drums, and one or more singers. 

  

[Blues] Blues is a music genre and musical form that originated among African Americans in the Deep South of the United States around the 1860s. Blues has incorporated spirituals, work songs, field hollers, shouts, chants, and rhymed simple narrative ballads from the African-American culture. The blues form is ubiquitous in jazz, rhythm and blues, and rock and roll, and is characterized by the call-and-response pattern, the blues scale, and specific chord progressions, of which the twelve-bar blues is the most common. Blue notes, usually thirds, fifths or sevenths flattened in pitch, are also an essential part of the sound. Blues shuffles or walking bass reinforce the trance-like rhythm and form a repetitive effect known as the groove. 

  

[Reggae] Reggae is a music genre that originated in Jamaica in the late 1960s. The term also refers to the modern popular music of Jamaica and its diaspora. The 1968 single by Toots and the Maytals titled "Do the Reggay" was the first popular song to use the word reggae, effectively naming the genre and introducing it to a global audience. 

  

[Electronic music] Electronic music broadly is a group of music genres that employ electronic musical instruments, circuitry-based music technology and software, or general-purpose electronics in its creation. It includes both music made using electronic and electromechanical means. Pure electronic instruments depend entirely on circuitry-based sound generation, for instance using devices such as an electronic oscillator, theremin, or synthesizer: no acoustic waves need to be previously generated by mechanical means and then converted into electrical signals. On the other hand, electromechanical instruments have mechanical parts such as strings or hammers that generate the sound waves, together with electric elements including magnetic pickups, power amplifiers and loudspeakers process the electric signals produced by the pickups and convert them back into sound waves. Such electromechanical devices include the telharmonium, Hammond organ, electric piano and electric guitar. 

  

[Film] A film, movie, or motion picture is a work of visual art that simulates experiences and otherwise communicates ideas, stories, perceptions, emotions, or atmosphere through the use of moving images that are generally, since the 1930s, synchronized with sound and sometimes feature other sensory stimuli. 

  

[Photography] Photography is the art, application, and practice of creating images by recording light, either electronically by means of an image sensor, or chemically by means of a light-sensitive material such as photographic film. It is employed in many fields of science, manufacturing, and business, as well as its more direct uses for art, film and video production, recreational purposes, hobby, and mass communication. A person who operates a camera to capture or take photographs is called a photographer, while the captured image, also known as a photograph, is the result produced by the camera. 

  

[Journalism] Journalism is the production and distribution of reports on events, facts, ideas, and people that constitute the "news of the day" and inform society with a commitment to accuracy and verification. The word, a noun, applies to the occupation, the methods of gathering information, and the organizing literary styles. 

  

[Animation] Animation is a filmmaking technique whereby pictures are created or manipulated and then played in sequence to create the illusion of moving images. In traditional animation, images are drawn or painted by hand on transparent celluloid sheets to be photographed and exhibited on film. Animation has been recognised as an artistic medium, specifically within the entertainment industry. Many animations are either traditional animations or computer animations made with computer-generated imagery (CGI). Stop motion animation, in particular claymation, is also prominent alongside these other forms, albeit to a lesser degree. 

  

[Video game] A video game, computer game, or simply game is an electronic game that involves interaction with a user interface or input device to generate visual feedback from a display device, most commonly shown in a video format on a television set, computer monitor, flat-panel display or touchscreen on handheld devices, or a virtual reality headset. Most modern video games are audiovisual, with audio complement delivered through speakers or headphones, and sometimes also with other types of sensory feedback. Some video games also allow microphone and webcam inputs for in-game chatting and livestreaming. 

  

[Printing press] A printing press is a machine that transfers ink onto materials such as paper or cloth by applying pressure to an inked surface. It marked a major advance on earlier methods, in which ink was applied to the printing surface and the paper was rubbed by hand to transfer it. The invention and global spread of the printing press transformed book production and communication during the early modern period. 

  

[Japan] Japan is an island country in East Asia. Located in the Pacific Ocean off the northeast coast of the Asian mainland, it is bordered to the west by the Sea of Japan, the Sea of Okhotsk in the north, and the East China Sea in the south. The Japanese archipelago consists of four major islands alongside over 14,000 smaller islands. Japan is divided into 47 administrative prefectures and eight traditional regions, and around 75% of its terrain is mountainous and heavily forested, concentrating its agriculture and highly urbanized population along its eastern coastal plains. With a population of almost 123 million as of 2026, it is the world's 11th most populous country. Tokyo is the country's capital and largest city. 

  

[Brazil] Brazil, officially the Federative Republic of Brazil, is the largest country in South America. It is also the world's fifth-largest country by area and the seventh-largest by population, with over 213 million people. Brazil is a federation composed of 26 states and a Federal District, which hosts the capital, Brasília. Its most populous city is São Paulo, followed by Rio de Janeiro. Brazil has the largest Lusophone population in the world and is the only Portuguese-speaking country in the Americas, where it is the official language. 

  

[India] India, officially the Republic of India, is a country in South Asia. It is the world's seventh-largest country by area and the largest by population. Bounded by the Indian Ocean on the south, the Arabian Sea on the southwest, and the Bay of Bengal on the southeast, it shares land borders with Pakistan to the west; China, Nepal, and Bhutan to the north; and Bangladesh and Myanmar to the east. In the Indian Ocean, India is near Sri Lanka and the Maldives. 

  

[Egypt] Egypt, officially the Arab Republic of Egypt, is a country spanning the northeast corner of Africa and southwest corner of Asia via the Sinai Peninsula. It is bordered by the Mediterranean Sea to the north, Palestine and Israel to the northeast, the Red Sea to the east, Sudan and the Sahara to the south, and Libya to the west. The Gulf of Aqaba in the northeast separates Egypt from Jordan and Saudi Arabia. Cairo is the capital, largest city, and leading cultural centre, while Alexandria is the second-largest city and an important hub of industry and tourism. With over 107 million inhabitants, Egypt is the most populous country in the Arab world, third-most populous country in Africa, and 15th-most populated in the world. 

  

[South Africa] South Africa, officially the Republic of South Africa (RSA), is the southernmost country in Africa. The country consists of nine provinces, and is bounded to the south by 2,798 kilometres of coastline that stretches along the South Atlantic and Indian Ocean; to the north by the neighbouring countries of Namibia, Botswana, and Zimbabwe; and to the east and northeast by Mozambique and Eswatini. South Africa also encloses Lesotho. Covering an area of 1,221,037 square kilometres, the country has a population of over 63 million people, making it the sixth-most populated country in Africa, and 24th-most in the world. 

  

[Australia] Australia, officially the Commonwealth of Australia, is a country comprising the mainland of the Australian continent, the island of Tasmania and numerous smaller islands. It has a land area of 7,688,287 km2 (2,968,464 sq mi), making it the sixth-largest country in the world, and is the world's flattest and driest inhabited continent. It is a megadiverse country, and its size gives it a wide variety of landscapes and climates, including deserts in the interior and tropical rainforests along the coast. 

  

[Canada] Canada is a country in North America. Its ten provinces and three territories extend from the Atlantic Ocean to the Pacific Ocean and northward into the Arctic Ocean, making it the second-largest country by total area, with the longest coastline of any country. Its border with the United States is the longest international land border. The country is characterized by a wide range of both meteorologic and geological regions. With a population of over 41 million, it has widely varying population densities, with the majority residing in its urban areas and large areas being sparsely populated. Its capital is Ottawa and its three largest metropolitan areas are Toronto, Montreal, and Vancouver. 

  

[France] France, officially the French Republic, is a country primarily located in Western Europe. Its overseas regions and territories include French Guiana in South America, Saint Pierre and Miquelon in the North Atlantic, the French West Indies, and many islands in Oceania and the Indian Ocean. Metropolitan France shares borders with Belgium and Luxembourg to the north; Germany to the northeast; Switzerland to the east; Italy and Monaco to the southeast; Andorra and Spain to the south; and a maritime border with the United Kingdom to the northwest. Its metropolitan area extends from the Rhine to the Atlantic Ocean and from the Mediterranean Sea to the English Channel and the North Sea. Its 18 integral regions—five of which are overseas—span a combined area of 632,702 km2 (244,288 sq mi), with a total population estimated at over 69.1 million in 2026. Its capital, largest city and main cultural and economic centre is Paris, with a metropolitan population of over 13 million. 

  

[Germany] Germany, officially the Federal Republic of Germany, is a country in Western and Central Europe. It lies between the Baltic Sea and the North Sea to the north with the Alps to the south. Its 16 constituent states have a total population of over 83 million, making it the most populous member state of the European Union (EU). Germany borders Denmark to the north; Poland and the Czech Republic to the east; Austria and Switzerland to the south; and France, Luxembourg, Belgium, and the Netherlands to the west. The nation's capital and most populous city is Berlin and its main financial centre is Frankfurt; the largest urban area is the Ruhr. 

  

[Mexico] Mexico, officially the United Mexican States, is a country in North America. It is the northernmost country in Latin America, bordering the United States of America to the north and Guatemala and Belize to the southeast, while having maritime boundaries with the Pacific Ocean to the west, the Caribbean Sea to the southeast, and the Gulf of Mexico to the east. Mexico covers 1,972,550 km2, and is the thirteenth-largest country in the world by land area. With a population exceeding 134 million as of 2026, Mexico is the tenth-most populous country in the world and is home to the largest number of native Spanish speakers. Mexico City is the capital and largest city in Mexico, which ranks among the most populous metropolitan areas in the world. 

  

[China] China, officially the People's Republic of China (PRC), is a country in East Asia. It is divided into 33 province-level divisions, including two special administrative regions. Beijing is the capital, while Shanghai is the most populous city by urban area. Its geography features the Central Plain, major rivers such as the Yangtze and Yellow River, deserts, subtropical and temperate forests, plateaus, and mountain ranges such as the Himalayas. It is the world's second-most populous country after India, with a population exceeding 1.4 billion, across an area of 9.6 million square kilometers (3,700,000 sq mi), making it the third-largest country by area. 

  

[Russia] Russia, or the Russian Federation, is a country in Eastern Europe and North Asia. It is the largest country in the world, spanning eleven time zones and sharing land borders with fourteen countries. With a population of over 140 million, Russia is the most populous country in Europe and the ninth-most populous in the world. It is a highly urbanised country, with sixteen of its urban areas having more than 1 million inhabitants. Moscow, the most populous metropolitan area in Europe, is the capital and largest city of Russia, while Saint Petersburg is its second-largest city and a major cultural centre. 

  

[Great Wall of China] The Great Wall of China is a series of fortifications in China. They were built across the historical northern borders of ancient Chinese states and Imperial China as protection against various nomadic groups from the Eurasian Steppe. The first walls date to the 7th century BC; these were joined together in the Qin dynasty. Successive dynasties expanded the wall system; the best-known sections were built by the Ming dynasty (1368–1644). 

  

[Egyptian pyramids] The Egyptian pyramids are ancient masonry structures located in Egypt. Most were built as tombs for the pharaohs and their consorts during the Old and Middle Kingdom periods. At least 138 identified pyramids have been discovered in Egypt. Approximately 80 pyramids were built within the Kingdom of Kush, now located in the modern country of Sudan. 

  

[Taj Mahal] The Taj Mahal is an ivory-white marble mausoleum on the right bank of the river Yamuna in Agra, Uttar Pradesh, India. It was commissioned in 1631 by the fifth Mughal emperor, Shah Jahan, to house the tomb of his late wife, Mumtaz Mahal; it also houses the tomb of Shah Jahan himself. The tomb is the centrepiece of a 17-hectare (42-acre) complex, which includes a mosque and a guest house, and is set in formal gardens bounded on three sides by a crenellated wall. 

  

[Colosseum] The Colosseum is an elliptical amphitheatre in the centre of the city of Rome, Italy, just east of the Roman Forum. It is the largest ancient amphitheatre ever built, and is the largest standing amphitheatre in the world. Construction began under the Emperor Vespasian in 72 and was completed in AD 80 under his successor and heir, Titus. Further modifications were made during the reign of Domitian. The three emperors who were patrons of the work are known as the Flavian dynasty, and the amphitheatre was named the Flavian Amphitheatre by later classicists and archaeologists for its association with their family name (Flavius). 

  

[Machu Picchu] Machu Picchu is a 15th-century Inca citadel located in the Eastern Cordillera of southern Peru on a mountain ridge at 2,430 meters (7,970 ft). It is situated in the Machupicchu District of Urubamba Province about 80 kilometers northwest of Cusco, above the Sacred Valley and along the Urubamba River, which forms a deep canyon with a subtropical mountain climate. 

  

[Grand Canyon] The Grand Canyon is a steep-sided canyon carved by the Colorado River in Arizona, United States. The Grand Canyon is 277 miles (446 km) long, up to 18 miles (29 km) wide and attains a depth of over a mile. 

  

[Mount Everest] Mount Everest is the highest mountain on Earth above sea level. It lies in the Mahalangur Himal sub-range of the Himalayas and marks part of the China–Nepal border at its summit. Its height was most recently measured in 2020 through a joint survey by Nepalese and Chinese authorities as 8,848.86 m. 

  

[Amazon rainforest] The Amazon rainforest, also called the Amazon jungle, Amazonia, or simply the Amazon, is a moist broadleaf tropical rainforest in the Amazon biome that covers most of the Amazon basin of South America. This basin encompasses 7 million km2 (2.7 million sq mi), of which 6 million km2 (2.3 million sq mi) are covered by the rainforest. This region includes territory belonging to nine nations and 3,344 indigenous territories. 

  

[Solar System] The Solar System is the gravitationally bound system of the Sun and the masses that orbit it, most prominently its eight planets, of which Earth is one. The Solar System is an isolated single-star planetary system within the Milky Way Galaxy. The system formed about 4.6 billion years ago when a dense region of a molecular cloud collapsed, creating the Sun and a protoplanetary disc from which the orbiting bodies assembled. 

  

[Sun] The Sun is the star located at the centre of the Solar System. It is a massive sphere of hot plasma, heated to incandescence by nuclear fusion reactions in its core, radiating the energy from its surface mainly as visible light and infrared radiation with 10% at ultraviolet energies. It is the main source of energy for life on Earth. The Sun has been an object of veneration in many cultures and a central subject of astronomical research since antiquity. 

  

[Mars] Mars is the fourth planet from the Sun. It is also known as the "Red Planet", for its orange-red appearance. Mars is a desert-like rocky planet with a tenuous atmosphere that is primarily carbon dioxide. At the average surface level the atmospheric pressure is a few thousandths of Earth's, atmospheric temperature ranges from −153 to 20 °C, and cosmic radiation is high. Mars retains some water, in the ground as well as thinly in the atmosphere, forming cirrus clouds, fog, frost, larger polar regions of permafrost and ice caps, but no bodies of liquid surface water. Its surface gravity is roughly a third of Earth's or double that of the Moon. Its mean diameter, 6,779 km (4,212 mi), is about half the Earth's, or twice the Moon's, and its surface area is the size of all the dry land of Earth. 

  

[Jupiter] Jupiter is the fifth planet from the Sun, and the largest in the Solar System. It is a gas giant with a mass nearly 2.5 times that of all the other planets in the Solar System combined and slightly less than one-thousandth the mass of the Sun. The diameter of Jupiter is 11 times that of Earth and a tenth that of the Sun. It orbits the Sun at a distance of 5.20 AU (778.5 Gm), with an orbital period of 11.86 years. Jupiter is the third-brightest natural object in the Earth's night sky, after the Moon and Venus, and has been observed since prehistoric times. Its name derives from that of Jupiter, the chief deity of ancient Roman religion. 

  

[Black hole] A black hole is an astronomical body so compact that its gravity prevents anything, including light, from escaping. Albert Einstein's theory of general relativity, which describes gravitation as the curvature of spacetime, predicts that any sufficiently compact mass will form a black hole. The boundary of no escape is called the event horizon. In general relativity, crossing a black hole's event horizon traps an object inside but produces no locally detectable change. General relativity also predicts that every black hole should have a central singularity, where the curvature of spacetime is infinite. 

  

[Milky Way] The Milky Way or Milky Way Galaxy, or simply the Galaxy, is the galaxy that includes the Solar System, with the name describing the galaxy's appearance from Earth: a hazy band of light seen in the night sky formed from stars in other arms of the galaxy, which are so far away that they cannot be individually distinguished by the naked eye. 

  

[Big Bang] The Big Bang is a physical theory that describes how the universe expanded from an early state of high density and temperature. Various cosmological models based on the Big Bang concept explain a broad range of phenomena, including the abundance of light elements, the cosmic microwave background (CMB) radiation, the redshift of galaxies and the large-scale structure of the universe. The observed uniformity of the universe, which leads to the horizon and flatness problems, is explained through cosmic inflation: a phase of accelerated expansion during the earliest stages. Detailed measurements of the expansion rate of the universe place the cosmic inflation at an estimated 13.787±0.02 billion years ago, which is considered the age of the universe. A wide range of empirical evidence strongly favors the Big Bang model, which is now widely accepted. 

  

[Dark matter] In astronomy and cosmology, dark matter is an invisible and hypothetical form of matter that does not interact with electromagnetic radiation, including light. Dark matter is implied by gravitational effects that cannot be explained by general relativity unless more matter is present than can be observed. Such effects occur in the context of formation and evolution of galaxies, gravitational lensing, the observable universe's current structure, mass position in galactic collisions, the motion of galaxies within galaxy clusters, and cosmic microwave background anisotropies. Dark matter is thought to serve as gravitational scaffolding for cosmic structures. After the Big Bang, dark matter clumped into blobs along narrow filaments with superclusters of galaxies forming a cosmic web at scales on which entire galaxies appear like tiny particles. 

  

[Dark energy] In physical cosmology and astronomy, dark energy is a proposed form of energy that affects the universe on its largest scales. Its primary effect is to drive the accelerating expansion of the universe. It also slows the rate of structure formation. Assuming that the lambda-CDM model of cosmology is correct, dark energy dominates the universe, contributing 68% of the total mass-energy in the present-day observable universe while dark matter and ordinary (baryonic) matter contribute 27% and 5%, respectively, and other components such as neutrinos and photons are nearly negligible. Dark energy's density is very low: 7×10−30 g/cm3, much lower than the density of ordinary matter or dark matter within galaxies. However, it dominates the universe's mass–energy content because it is uniform across space. 

  

[Exoplanet] An exoplanet or extrasolar planet is a planet outside the Solar System. The first confirmed detection of an exoplanet was in 1992 around a pulsar, and the first detection around a main-sequence star was in 1995. A different planet, first detected in 1988, was confirmed in 2003. In 2016, it was recognized that the first possible evidence of an exoplanet had been noted in 1917, a precovery. As of 20 August 2026, there are 6,354 confirmed exoplanets in 4,756 planetary systems, with 1,060 systems having more than one planet. 

  

[James Webb Space Telescope] The James Webb Space Telescope (JWST) is a space telescope designed to conduct infrared astronomy. It is the largest telescope in space, and is equipped with high-resolution and high-sensitivity instruments, allowing it to view objects too old, distant, or faint for the Hubble Space Telescope. This enables investigations across many fields of astronomy and cosmology, such as observation of the first stars and the formation of the first galaxies, and detailed atmospheric characterization of potentially habitable exoplanets. 

  

[International Space Station] The International Space Station (ISS) is a space station in low Earth orbit (LEO). It is the product of the International Space Station program and is operated by five partner space agencies: NASA, Roscosmos (Russia), ESA (Europe), JAXA (Japan), and CSA (Canada). It is the first space station built, maintained and crewed through international cooperation and the largest human spacecraft ever constructed. It is an orbital research station, where scientific experiments in microgravity are conducted and the space environment is studied. Since 2 November 2000, it has hosted the longest continuous human presence in space. Alongside China's Tiangong, it is one of two operational space stations. 

  

[Apollo program] The Apollo program, also known as Project Apollo, was the United States human spaceflight program led by NASA, which landed the first humans on the Moon in 1969. Apollo was conceived in 1960 in the Dwight D. Eisenhower presidency during Project Mercury and executed after Project Gemini. Apollo was later dedicated to President John F. Kennedy's national goal, "before this decade is out, of landing a man on the Moon and returning him safely to the Earth" in his address to the U.S. Congress on May 25, 1961. 

  

[Supernova] A supernova is a powerful and luminous explosion of a star. A supernova occurs during the last evolutionary stages of a massive star, or when a white dwarf is triggered into runaway nuclear fusion. The original object, called the progenitor, either collapses to a neutron star or black hole, or is completely destroyed to form a diffuse nebula. The peak optical luminosity of a supernova can be comparable to that of an entire galaxy before fading over several weeks or months. 

  

[Electromagnetism] In physics, electromagnetism is an interaction that occurs between particles with electric charge via electromagnetic fields. The electromagnetic force is one of the four fundamental forces of nature. It is the dominant force in the interactions of atoms and molecules. Electromagnetism describes and relates the three distinct but closely intertwined phenomena of electricity, magnetism, and optics. In the study of electromagnetism these phenomena are described by the 3 sub-disciplines: electrostatics, magnetostatics, and electrodynamics. 

  

[Thermodynamics] Thermodynamics is a branch of physics that deals with heat, work, and temperature, and their relation to energy, entropy, and the physical properties of matter and radiation. The behavior of these quantities is governed by the four laws of thermodynamics, which convey a quantitative description using measurable macroscopic physical quantities but may be explained in terms of microscopic constituents by statistical mechanics. Thermodynamics applies to various topics in science and engineering, especially physical chemistry, biochemistry, chemical engineering, and mechanical engineering, as well as other complex fields such as meteorology. 

  

[Special relativity] In physics, the special theory of relativity, or simply special relativity, is a scientific theory of the relationship between space and time. In Albert Einstein's 1905 paper,  

"On the Electrodynamics of Moving Bodies", the theory is presented as being based on just two postulates:The laws of physics are invariant (identical) in all inertial frames of reference. This is known as the principle of relativity. 

The speed of light in vacuum is the same for all observers, regardless of the motion of light source or observer. This is known as the principle of light constancy, or the principle of light speed invariance. 

  

[Standard Model] The Standard Model of particle physics is the theory describing three of the four known fundamental forces in the universe and classifying all known elementary particles. It was developed in stages throughout the latter half of the 20th century, through the work of many scientists worldwide, with the current formulation being finalized in the mid-1970s upon experimental confirmation of the existence of quarks. Since then, proof of the top quark (1995), the tau neutrino (2000), and the Higgs boson (2012) have added further credence to the Standard Model. In addition, the Standard Model has predicted with great accuracy the various properties of weak neutral currents and the W and Z bosons. 

  

[Higgs boson] The Higgs boson, sometimes called the Higgs particle, is an elementary particle in the Standard Model of particle physics produced by the quantum excitation of the Higgs field, one of the fields in particle physics theory. In the Standard Model, the Higgs particle is a massive scalar boson that couples to particles whose mass arises from their interactions with the Higgs field, has zero spin, even (positive) parity, no electric charge, and no color charge. It is also very unstable, decaying into other particles almost immediately upon generation. 

  

[Gravitational wave] Gravitational waves are waves of spacetime curvature produced by the relative motion of gravitating masses and which propagate away at the speed of light. They were first predicted by Albert Einstein as a consequence of his general theory of relativity, appearing as "ripples in spacetime curvature". 

Hundreds of these gravitational waves have since then been observed, first indirectly using binary-pulsar observations and, since 2015, directly through dedicated observatories. 

  

[Entropy] Entropy is a thermodynamic state variable that quantifies the probabilistic distribution of accessible microstates in a system. The term and the concept are used in diverse fields, from classical thermodynamics, to the microscopic description of nature in statistical physics, and the principles of information theory. It has far-ranging applications in chemistry and physics, in biological systems and their relation to life, in cosmology, economics, and information systems including the transmission of information in telecommunication. 

  

[Nuclear fusion] Nuclear fusion is a reaction in which two or more atomic nuclei combine to form a larger nucleus. The difference in mass between the reactants and products is manifested as either the release or the absorption of energy. This difference in mass arises as a result of the difference in nuclear binding energy between the atomic nuclei before and after the fusion reaction. Active stellar cores are powered by fusion. Nucleosynthesis via fusion, in the Big Bang and in stars, creates all elements lighter than nickel. 

  

[Periodic table] The periodic table, also known as the periodic table of the elements, is an ordered arrangement of the chemical elements into rows ("periods") and columns ("groups"). An icon of chemistry, the periodic table is widely used in physics and other sciences. It is a depiction of the periodic law, which states that when the elements are arranged in order of their atomic numbers an approximate recurrence of their properties is evident. The table is divided into four roughly rectangular areas called blocks. Elements in the same group tend to show similar chemical characteristics. 

  

[Organic chemistry] Organic chemistry is a subdiscipline within chemistry involving the scientific study of the structure, properties, and reactions of organic compounds and organic materials. It involves studying the structure of organic material to determine the structural formula, analyzing physical and chemical properties, and evaluating chemical reactivity to understand the behavior of organic compounds. The study of organic reactions includes the chemical synthesis of natural products, drugs, and polymers, and study of individual organic molecules in the laboratory and via theoretical study. 

  

[Polymer] A polymer is a substance or material that consists of very large molecules, or macromolecules, that are constituted by many repeating subunits derived from one or more species of monomers. Due to their broad spectrum of properties, both synthetic and natural polymers play essential and ubiquitous roles in everyday life. Polymers range from familiar synthetic plastics such as polystyrene to natural biopolymers such as DNA and proteins that are fundamental to biological structure and function. Polymers, both natural and synthetic, are created via polymerization of many small molecules, known as monomers. Their consequently large molecular mass, relative to small molecule compounds, produces unique physical properties including toughness, high elasticity, viscoelasticity, and a tendency to form amorphous and semicrystalline structures rather than crystals. 

  

[Catalysis] Catalysis is the increase in rate of a chemical reaction due to an added substance known as a catalyst. Catalysts are not consumed by the reaction and remain unchanged after the reaction. If the reaction is rapid and the catalyst is recycled quickly, a very small amount of catalyst often suffices; mixing, surface area, and temperature are important factors in reaction rate. Catalysts generally react with one or more reactants to form intermediates that subsequently give the final reaction product, in the process of regenerating the catalyst. 

  

[Electrochemistry] Electrochemistry is the branch of physical chemistry concerned with the relationship between electrical potential difference and identifiable chemical change. These reactions involve electrons moving via an electronically conducting phase between electrodes separated by an ionically conducting and electronically insulating electrolyte. The specialization of electrochemistry in the nanoscale is called nanoelectrochemistry. 

  

[Chemical bond] A chemical bond is the association of atoms or ions to form molecules, crystals, and other structures. The bond may result from the electrostatic force between oppositely charged ions, as in ionic bonds; the sharing of electrons, as in covalent bonds; or some combination of these effects. Chemical bonds are described as having different strengths: there are "strong bonds" or "primary bonds" such as covalent, ionic, and metallic bonds, and "weak bonds" or "secondary bonds" such as dipole–dipole interactions, the London dispersion force, and hydrogen bonds. 

  

[Cell (biology)] The cell is the basic structural and functional unit of all forms of life or organisms. The term comes from the Latin word cellula meaning 'small room'. A biological cell basically consists of a semipermeable cell membrane enclosing cytoplasm that contains genetic material. Most cells are only visible under a microscope. Except for highly-differentiated cell types most cells are capable of replication, and protein synthesis. Some types of cell are motile. Cells emerged on Earth about four billion years ago. 

  

[DNA] Deoxyribonucleic acid is a polymer composed of two polynucleotide chains that coil around each other to form a double helix. The polymer carries genetic instructions for the development, functioning, growth and reproduction of all known organisms and many viruses. DNA and ribonucleic acid (RNA) are nucleic acids. Alongside proteins, lipids and complex carbohydrates (polysaccharides), nucleic acids are one of the four major types of macromolecules that are essential for all known forms of life. 

  

[Photosynthesis] Photosynthesis is a system of biological processes by which photopigment-bearing autotrophic organisms, such as most plants, algae and cyanobacteria, convert light energy—typically from sunlight—into the chemical energy necessary to fuel their metabolism. The term photosynthesis usually refers to oxygenic photosynthesis, a process that releases oxygen as a byproduct of water splitting. Photosynthetic organisms store the converted chemical energy within the bonds of intracellular organic compounds, typically carbohydrates like sugars, starches, phytoglycogen and cellulose. When needing to use this stored energy, an organism's cells then metabolize the organic compounds through cellular respiration. Photosynthesis plays a critical role in producing and maintaining the oxygen content of the Earth's atmosphere, and it supplies most of the biological energy necessary for complex life on Earth. 

  

[Immune system] The immune system is a network of biological systems that protects an organism from diseases. It detects and responds to a wide variety of pathogens, such as viruses, bacteria, and parasites, as well as cancer cells and foreign objects, such as wood splinters—distinguishing them from the organism's own healthy tissue. Many species have two major subsystems of the immune system. The innate immune system provides a preconfigured response to broad groups of situations and stimuli. The adaptive immune system provides a tailored response to each stimulus by learning to recognize molecules it has previously encountered. Both use molecules and cells to perform their functions. 

  

[Virus] A virus is a submicroscopic infectious agent that replicates only inside the living cells of an organism. Viruses infect all life forms, from animals and plants to microorganisms, including bacteria and archaea. Viruses are found in almost every ecosystem on Earth and are the most numerous type of biological entity. Since the first discovery of a virus, the tobacco mosaic virus, in the 1890s, more than 16,000 of the millions of virus species have been described in detail. The study of viruses is known as virology, a subspeciality of microbiology. 

  

[Bacteria] Bacteria are ubiquitous, mostly free-living organisms often consisting of one biological cell. They constitute a large domain of prokaryotic microorganisms. Typically a few micrometres in length, bacteria were among the first life forms to appear on Earth, and are present in most of its habitats. Bacteria inhabit the air, soil, water, acidic hot springs, radioactive waste, and the deep biosphere of Earth's crust. Bacteria play a vital role in many stages of the nutrient cycle by recycling nutrients and the fixation of nitrogen from the atmosphere. The nutrient cycle includes the decomposition of dead bodies; bacteria are responsible for the putrefaction stage in this process. In the biological communities surrounding hydrothermal vents and cold seeps, extremophile bacteria provide the nutrients needed to sustain life by converting dissolved compounds, such as hydrogen sulphide and methane, to energy. Bacteria also live in mutualistic, commensal and parasitic relationships with plants and animals. Most bacteria have not been characterised and there are many species that cannot be grown in the laboratory. The study of bacteria is known as bacteriology, a branch of microbiology. 

  

[Fungus] A fungus is any member of the group of eukaryotic organisms that includes yeasts, molds, and mushrooms. These organisms are classified in the biological kingdom Fungi. 

  

[Natural selection] Natural selection is the differential survival and reproduction of individuals due to differences in the relative fitness endowed on them by their own particular complement of observable characteristics. It is a key law or mechanism of evolution which changes the heritable traits characteristic of a population or species over generations. Charles Darwin popularised the term "natural selection", contrasting it with artificial selection, which is intentional, whereas natural selection is not. 

  

[On the Origin of Species] On the Origin of Species by Means of Natural Selection, or the Preservation of Favoured Races in the Struggle for Life is a work of scientific literature by the English naturalist Charles Darwin that is considered to be the foundation of evolutionary biology. It was published on 24 November 1859. Darwin's book introduced the scientific theory that populations evolve over the course of generations through a process of natural selection, although Lamarckism was also included as a mechanism of lesser importance. The book presented a body of evidence that the diversity of life arose by common descent through a branching pattern of evolution. Darwin included evidence that he had collected on the Beagle expedition in the 1830s and his subsequent findings from research, correspondence, and experimentation. 

  

[Protein] Proteins are large biomolecules and macromolecules that comprise one or more long chains of amino acid residues. Proteins perform a vast array of functions within organisms, including catalysing metabolic reactions, DNA replication, responding to stimuli, providing structure to cells and organisms, and transporting molecules from one location to another. Proteins differ from one another primarily in their sequence of amino acids, which is dictated by the nucleotide sequence of their genes, and which usually results in protein folding into a specific 3D structure that determines its activity. 

  

[Elephant] Elephants are the largest living land animals. Three living species are currently recognised: the African bush elephant, the African forest elephant, and the Asian elephant. They are the only surviving members of the family Elephantidae and the order Proboscidea; extinct relatives include mammoths and mastodons. Distinctive features of elephants include a long proboscis called a trunk, tusks, large ear flaps, pillar-like legs, and tough but sensitive grey skin. The trunk is prehensile, bringing food and water to the mouth and grasping objects. Tusks, which are derived from the incisor teeth, serve both as weapons and as tools for moving objects and digging. The large ear flaps assist in maintaining a constant body temperature as well as in communication. African elephants have larger ears and concave backs, whereas Asian elephants have smaller ears and convex or level backs. 

  

[Lion] The lion is a large cat of the genus Panthera, currently ranging only in Sub-Saharan Africa and India. It has a muscular, broad-chested body; a short, rounded head; round ears; and a dark, hairy tuft at the tip of its tail. It is sexually dimorphic; adult male lions are larger than females and have a prominent mane that extends from the head to the shoulders and chest. 

  

[Tiger] The tiger is a large cat and a member of the genus Panthera native to Asia. It has a powerful, muscular body with a large head and paws, a long tail and orange fur that fades to white in parts, with black, mostly vertical stripes. It is traditionally classified into nine subspecies, though some recognise only two subspecies, the mainland Asian tiger and the Sunda Islands tiger. 

  

[Blue whale] The blue whale is a species of baleen whale and the largest marine mammal in the rorqual family Balaenopteridae. Reaching a maximum confirmed length of 29.9–30.5 m (98–100 ft) and weighing up to 190–200 t, it is the largest animal known to have ever existed. The blue whale's long and slender body can be of various shades of greyish-blue on its upper surface and somewhat lighter underneath. Four subspecies are recognized: B. m. musculus in the North Atlantic and North Pacific, B. m. intermedia in the Southern Ocean, B. m. brevicauda in the Indian Ocean and South Pacific Ocean, and B. m. indica in the Northern Indian Ocean. There is a population in the waters off Chile that may constitute a fifth subspecies. 

  

[Octopus] An octopus is a soft-bodied, eight-limbed mollusc of the order Octopoda. The order consists of some 300 species and is grouped within the class Cephalopoda with squids, cuttlefish, and nautiloids. Like other cephalopods, an octopus is bilaterally symmetric with two eyes and a beaked mouth at the centre point of the eight limbs. An octopus can radically deform its shape, enabling it to squeeze through small gaps. They trail their appendages behind them as they swim backwards. The siphon is used for respiration and locomotion. Octopuses have a complex nervous system and excellent sight, and are among the most intelligent and behaviourally diverse invertebrates. 

  

[Dolphin] A dolphin is any one of the 40 extant species of aquatic mammal from the cetacean families Delphinidae, Platanistidae, Iniidae, Pontoporiidae, and the probably extinct Lipotidae. All these families belong to the parvorder Odontoceti, i.e., toothed whales, which also include the closely related families Monodontidae and Phocoenidae (porpoises), as well as the more distant families Physeteroidea and Ziphiidae. 

  

[Penguin] Penguins are a group of flightless semi-aquatic sea birds which live almost exclusively in the Southern Hemisphere. Only one species, the Galapagos penguin, lives at, and slightly north of, the equator. Highly adapted for life in the ocean water, penguins have countershaded dark and white plumage and flippers for swimming. Most penguins feed on krill, fish, squid and other forms of sea life which they catch with their bills and swallow whole while swimming. A penguin has a spiny tongue and powerful jaws to grip slippery prey. 

  

[Giant panda] The giant panda, also known as the panda bear or simply panda, is a bear species endemic to China. It is characterised by its white coat with black patches around the eyes, ears, legs and shoulders. Its body is rotund; adult individuals weigh 100 to 115 kg and are typically 1.2 to 1.9 m long. It is sexually dimorphic, with males being typically 10–20% larger than females. A thumb is visible on its forepaw, which helps in holding bamboo in place for feeding. It has large molar teeth and expanded temporal fossae to meet its dietary requirements. It can digest starch and is mostly herbivorous with a diet consisting almost entirely of bamboo and bamboo shoots. 

  

[Wolf] The wolf, also known as the grey wolf or gray wolf, is a canine native to Eurasia and North America. More than thirty subspecies of Canis lupus have been recognized, including the dog and dingo, though grey wolves, as popularly understood, include only naturally occurring wild subspecies. The wolf is the largest wild extant member of the family Canidae, and is further distinguished from other Canis species by its less pointed ears and muzzle, as well as a shorter torso and a longer tail. The wolf is nonetheless related closely enough to smaller Canis species, such as the coyote and the golden jackal, to produce fertile hybrids with them. The wolf's fur is usually mottled white, brown, grey, and black, although subspecies in the arctic region may be nearly all white. 

  

[Orca] The orca, or killer whale, is a toothed whale and the largest member of the oceanic dolphin family. The only extant species in the genus Orcinus, it is recognizable by its distinct pigmentation; being mostly black on top, white on the bottom and having recognizable white eye patches. A cosmopolitan species, it inhabits a wide range of marine environments, from Arctic to Antarctic regions to tropical seas, but is more commonly documented in temperate or cooler coastal waters. Scientists have proposed dividing the global population into races, subspecies, or possibly even species. 

  

[Cancer] Cancer is a group of diseases involving uncontrolled cell growth typically resulting in tumors with the potential to invade or spread to other parts of the body. These malignant tumors contrast with benign tumors, which do not spread. Over 100 types of cancers affect humans. 

  

[Diabetes] Diabetes mellitus, commonly known as diabetes or diabetus, is a group of common endocrine diseases characterized by sustained high blood sugar levels. Diabetes tends to progress in severity, and is due to either a reduced production of the hormone insulin by the pancreas or unresponsiveness of bodily cells to insulin's effects. Classic symptoms include the three Ps: polydipsia, polyuria, and polyphagia, together with weight loss and blurred vision. If left untreated, the disease can lead to many health complications, including disorders of the cardiovascular system, eye, kidney, and nerves. 

  

[Alzheimer's disease] Alzheimer's disease (AD) is a neurodegenerative disease and is the most common cause of dementia, accounting for around 60–70% of cases. The most common early symptom is difficulty in remembering recent events. As the disease advances, symptoms can include problems with language, disorientation, mood swings, loss of motivation, self-neglect, and behavioral issues. As a person's condition declines, they often withdraw from family and society. Gradually, bodily functions are lost, ultimately leading to death. The median life expectancy following diagnosis of dementia is three to twelve years. Co-occurring movement disorder, also known as extrapyramidal signs, multiplies risk of death by 1.6. 

  

[Malaria] Malaria is a mosquito-borne infectious disease that is transmitted by the bite of Anopheles mosquitoes. The symptoms of human malaria typically include fever, fatigue, vomiting, and headaches. In severe cases, the disease can cause jaundice, seizures, coma, or death. Symptoms usually begin 10 to 15 days after being bitten by an infected Anopheles mosquito. If not properly treated, people may have recurrences of the disease months later. Those who survive an infection develop partial immunity, being susceptible to reinfection although with milder symptoms. This partial resistance disappears over months to years if the person has no continuing exposure to malaria. 

  

[HIV/AIDS] The human immunodeficiency virus (HIV) is a retrovirus that attacks the immune system. Without treatment, it can lead to a spectrum of conditions including acquired immunodeficiency syndrome (AIDS). It is a preventable disease. It can be managed with treatment and become a manageable chronic health condition. While there is no cure or vaccine for HIV, antiretroviral treatment can slow the course of the disease, and, if used before significant disease progression, can extend the life expectancy of someone living with HIV to a nearly standard level. An HIV-positive person on treatment can expect to live a normal life, and die with the virus, not of it. Effective treatment for HIV-positive people involves a life-long regimen of medicine to suppress the virus, making the viral load undetectable. Early testing can show if treatment is needed to stop progression and to prevent infecting others. 

  

[Antibiotic] An antibiotic is a type of antimicrobial substance which is active against bacteria. It is the most important type of antibacterial agent for fighting bacterial infections, and antibiotic medications are widely used in the treatment and prevention of such infections. They may either kill or inhibit the growth of bacteria. A limited number of antibiotics also possess antiprotozoal activity, but antibiotics are not effective against viruses or fungi. 

  

[Anesthesia] Anesthesia or anaesthesia is a state of controlled, temporary loss of sensation or awareness that is induced for medical or veterinary purposes. It may include some or all of analgesia, paralysis, amnesia, and unconsciousness. An individual under the effects of anesthetic drugs is referred to as being anesthetized. 

  

[Gene therapy] Gene therapy is medical technology that aims to produce a therapeutic effect through the manipulation of gene expression or through altering the biological properties of living cells. 

  

[CRISPR] CRISPR is a family of DNA sequences found in the genomes of prokaryotic organisms such as bacteria and archaea. Each sequence within an individual prokaryotic CRISPR is derived from a DNA fragment of a bacteriophage that had previously infected the prokaryote or one of its ancestors. These sequences are used to detect and destroy DNA from similar bacteriophages during subsequent infections. Hence these sequences play a key role in the antiviral defense system of prokaryotes and provide a form of heritable, acquired immunity. CRISPR is found in approximately 50% of sequenced bacterial genomes and nearly 90% of sequenced archaea. 

  

[Major depressive disorder] Major depressive disorder (MDD), also known as clinical depression, is a mental disorder characterized by at least two weeks of pervasive low mood, low self-esteem, and loss of interest or pleasure in normally enjoyable activities. Introduced by a group of US clinicians in the mid-1970s, the term was adopted by the American Psychiatric Association for this symptom cluster under mood disorders in the 1980 version of the Diagnostic and Statistical Manual of Mental Disorders (DSM-III), and has become widely used since. The disorder causes the second-most years lived with disability, after lower back pain. 

  

[Transistor] A transistor is a semiconductor device used to amplify or switch electrical signals and power. It is one of the basic building blocks of modern electronics. It is composed of semiconductor material, usually with at least three terminals for connection to an electronic circuit. A voltage or current applied to one pair of the transistor's terminals controls the current through another pair of terminals. Because the controlled (output) power can be higher than the controlling (input) power, a transistor can amplify a signal. Some transistors are packaged individually, but many more in miniature form are found embedded in integrated circuits. Because transistors are the key active components in practically all modern electronics, they are considered one of the 20th century's greatest inventions. 

  

[Semiconductor] A semiconductor is a material with electrical conductivity between that of a conductor and an insulator. Its conductivity can be modified by adding impurities ("doping") to its crystal structure. When two regions with different doping levels are present in the same crystal, they form a semiconductor junction. The term "semiconductors" is sometimes used to refer to semiconductor devices such as microchips and computer processors, which work using the physical properties of semiconductors. 

  

[World Wide Web] The World Wide Web is a global interconnected information system that enables content sharing over the Internet. It facilitates access to documents and other web resources according to specific rules of the Hypertext Transfer Protocol (HTTP). 

  

[JavaScript] JavaScript (JS) is a programming language and core technology of the Web, alongside HTML and CSS. Created by Brendan Eich in 1995, it is maintained by Ecma International's TC39 technical committee, with related Web APIs maintained by W3C and WHATWG. As of 2025, JavaScript is the most widely used programming language on GitHub., however, TypeScript, which is a version of JavaScript that is strict about types and is statically typed, overtook JavaScript on Github by August 2025. 

  

[Linux] Linux is a family of free and open-source software Unix-like operating systems based on the Linux kernel, which was first released on 17 September 1991 by Linus Torvalds. Some members of the family are typically packaged as a distribution, which includes the kernel alongside supporting system software and libraries developed by third parties—such as GNU, Red Hat, and X.Org—to create a complete operating system; however, not all Linux-based operating systems are considered distros, with Android being an example. Linux was originally designed as a clone of Unix and is distributed under the copyleft GPL license. 

  

[Moore's law] Moore's law is the observation that the number of transistors in an integrated circuit (IC) doubles about every two years, with minimal increase in cost. Despite the name, Moore's law describes an empirical relationship, not a scientific law. This type of observation, an experience curve effect, quantifies efficiency gains from learned experience in production. 

  

[Algorithm] In mathematics and computer science, an algorithm is a finite sequence of mathematically rigorous logical instructions, typically used to solve a class of specific problems or to perform a computation. Algorithms are used as specifications for performing calculations and data processing. More advanced algorithms can use conditionals to divert the code execution through various routes and deduce valid inferences. 

  

[Global Positioning System] The Global Positioning System (GPS) is a satellite-based hyperbolic navigation system owned by the United States Space Force and operated by Mission Delta 31. It is one of the global navigation satellite systems (GNSS) that provide geolocation and time information to a GPS receiver anywhere on or near the Earth where signal quality permits. It does not require the user to transmit any data, and operates independently of any telephone or Internet reception, though these technologies can enhance the usefulness of the GPS positioning information. It provides critical positioning capabilities to military, civil, and commercial users around the world. Although the United States government created, controls, and maintains GPS, it is freely accessible to anyone with a GPS receiver. 

  

[Turing machine] A Turing machine is a mathematical model of computation describing an abstract machine that manipulates symbols on a strip of tape according to a table of rules. Despite the model's simplicity, it is capable of implementing any computer algorithm. 

  

[Integrated circuit] An integrated circuit (IC), also known as a microchip or simply chip, is a compact assembly of electronic circuits formed from various electronic components, such as transistors, resistors, and capacitors, and their interconnections. These components are fabricated onto a thin, flat piece ("chip") of semiconductor material, most commonly silicon. Integrated circuits are integral to a wide variety of electronic devices performing functions such as data processing, control, and storage. They have transformed the field of electronics by enabling device miniaturization, improving performance, and reducing cost. 

  

[Nuclear power] Nuclear power is the use of nuclear reactions to produce electricity. Nuclear power can be obtained from nuclear fission, nuclear decay and nuclear fusion reactions. Presently, the vast majority of electricity from nuclear power is produced by nuclear fission of uranium and plutonium in nuclear power plants. Nuclear decay processes are used in niche applications such as radioisotope thermoelectric generators in some space probes such as Voyager 2. Reactors producing controlled fusion power have been operated since 1958 but have yet to generate net power and are not expected to be commercially available in the near future. 

  

[Wind power] Wind power is the use of wind energy to generate useful work. Historically, wind power was used by sails, windmills and windpumps, but today it is mostly used to generate electricity. This article deals only with wind power for electricity generation. 

Today, wind power is generated almost completely using wind turbines, generally grouped into wind farms and connected to the electrical grid. 

  

[Desalination] Desalination is the artificial process by which salt water is converted to fresh water. More generally, desalination is the removal of salts and minerals from a substance. It is possible to desalinate saltwater, especially sea water, to produce water for human consumption or irrigation, producing brine as a by-product. 

  

[Biodiversity] Biodiversity is the variability of life on Earth. It can be measured on various levels, for example, genetic variability, species diversity, ecosystem diversity, and phylogenetic diversity. Diversity is not distributed evenly on Earth — it is greater in the tropics as a result of the warm climate and high primary productivity in the region near the equator. Tropical forest ecosystems cover less than one-fifth of Earth's terrestrial area and contain about 50% of the world's species. There are latitudinal gradients in species diversity for both marine and terrestrial taxa. 

  

[Greenhouse effect] The greenhouse effect occurs when heat-trapping gases in a planet's atmosphere prevent the planet from losing heat to space, raising its surface temperature. Surface heating can happen from an internal heat source or come from an external source, such as a host star. In the case of Earth, the Sun emits shortwave radiation (sunlight) that passes through greenhouse gases to heat the Earth's surface. In response, the Earth's surface emits longwave radiation that is mostly absorbed by greenhouse gases, reducing the rate at which the Earth can cool off. 

  

[Deforestation] Deforestation or forest clearance is the removal and destruction of a forest or stand of trees from land that is then converted to non-forest use. Deforestation can involve conversion of forest land to farms, ranches, or urban use. About 31% of Earth's land surface is covered by forests at present. This is one-third less than the forest cover before the expansion of agriculture, with half of that loss occurring in the last century. On average 2,400 trees are cut down each minute. Estimates vary widely as to the extent of deforestation in the tropics. In 2019, nearly a third of the overall tree cover loss, or 3.8 million hectares, occurred within humid tropical primary forests. These are areas of mature rainforest that are especially important for biodiversity and carbon storage. 

  

[Ozone layer] The ozone layer or ozone shield is a region of Earth's stratosphere that absorbs most of the Sun's ultraviolet radiation. It contains a high concentration of ozone (O3) in relation to other parts of the atmosphere, although still small in relation to other gases in the stratosphere. The ozone layer peaks at 8 to 15 parts per million of ozone, while the average ozone concentration in Earth's atmosphere as a whole is about 0.3 parts per million. The ozone layer is mainly found in the lower portion of the stratosphere, from approximately 15 to 35 kilometers (9 to 22 mi) above Earth, although its thickness varies seasonally and geographically. 

  

[Carbon cycle] The carbon cycle is a part of the biogeochemical cycle where carbon is exchanged among the biosphere, pedosphere, geosphere, hydrosphere, and atmosphere of Earth. Other major biogeochemical cycles include the nitrogen cycle and the water cycle. Carbon is the main component of biological compounds as well as a major component of many rocks such as limestone. The carbon cycle comprises a sequence of events that are key to making Earth capable of sustaining life. It describes the movement of carbon as it is recycled and reused throughout the biosphere, as well as long-term processes of carbon sequestration (storage) to and release from carbon sinks. At 422.7 parts per million (ppm), the global average atmospheric carbon dioxide has set a new record high in 2024. 

  

[Olympic Games] The modern Olympic Games are a series of international multi-sport events. Considered among the biggest sporting events in the world, the Olympics are organised into distinct Summer and Winter Games, each featuring different sports. The Olympic Games, open to both amateur and professional athletes, involve thousands of athletes and more than 200 teams, each team representing a sovereign state or territory. The Games often, but not always, substitute for any world championships during the year in which they take place. The Summer and Winter Olympics are each held every four years, and themselves occur two years apart within the four-year Olympiad. This arrangement has been in place since the 1994 Winter Olympics, prior to which the Summer and Winter Games were held in the same year. 

  

[Basketball] Basketball is a team sport in which two teams of five players each oppose one another on a rectangular court. Players compete with the primary objective of shooting a basketball through a hoop at each end of the court. Teams alternate between offense, and defense. A field goal is worth two points, unless made from behind the three-point line, when it is worth three. After a foul, timed play stops and the fouled team either takes one to three free throws worth one point each, or restarts play with an inbound pass. The team with the most points at the end of the game wins. If regulation play expires with the score tied, most basketball leagues mandate additional periods of play (overtime) until the score is no longer tied. 

  

[Tennis] Tennis is a racket sport that is played either individually against a single opponent (singles) or between two teams of two players each (doubles). Each player uses a tennis racket strung with a cord to strike a hollow rubber ball covered with felt over or around a net and into the opponent's court. The objective is to manoeuvre the ball in such a way that the opponent is not able to play a valid return. If a player is unable to return the ball successfully, the opponent scores a point. 

  

[Cricket] Cricket is a bat-and-ball game that is played between two teams of eleven players on a field, at the centre of which is a 22-yard pitch with a wicket at each end, each comprising two bails balanced on three stumps. Two players from the batting team, the striker and nonstriker, stand in front of either wicket holding bats, while one player from the fielding team, the bowler, bowls the ball toward the striker's wicket from the opposite end of the pitch. The striker's goal is to hit the bowled ball with the bat and then switch places with the nonstriker, with the batting team scoring one run for each of these swaps. Runs are also scored when the ball reaches the boundary of the field or when the ball is bowled illegally. 

  

[Cognitive bias] A cognitive bias is a systematic pattern of deviation from the norm or rationality in judgment. Individuals create their own "subjective reality" from their perception of the input. An individual's construction of reality, not the objective input, may dictate their behavior in the world. Thus, cognitive biases may sometimes lead to perceptual distortion, inaccurate judgment, illogical interpretation, and irrationality. 

  

[Memory] Memory is the faculty of the mind by which data or information is encoded, stored, and retrieved when needed. It is the retention of information over time for the purpose of influencing future action. If past events could not be remembered, it would be impossible for language, relationships, or personal identity to develop. Memory loss is usually described as forgetfulness or a disorder such as amnesia. 

  

[Emotion] Emotions are physical and mental states brought on by neurophysiological and neuropsychological changes, variously associated with subjective experiences such as thoughts, feelings, behavioral responses, and a degree of pleasure or displeasure. There is no scientific consensus on a definition. Emotions are often intertwined with mood, temperament, personality, disposition, or creativity. 

  

[Consciousness] Consciousness is being aware of something internal to one's self, or of states or objects in one's external environment. It has been the topic of extensive explanations, analyses, and debate among philosophers, scientists, and theologians for millennia. There is no consensus on what exactly needs to be studied, or whether consciousness can be considered a scientific concept. In some explanations it is synonymous with mind, while in others it is considered an aspect of it. 

  

[Human rights] Human rights are universally recognized moral principles or norms that establish standards of human behavior and are often protected by both national and international laws. These rights are considered inherent and inalienable, meaning they belong to every individual simply by virtue of being human. They encompass a broad range of civil, political, economic, social, and cultural rights, such as the right to life, freedom of speech, protection against enslavement, and right to education. 

  

[United Nations] The United Nations (UN) is an international intergovernmental organization established by the signing of the Charter of the United Nations on 26 June 1945 with the articulated mission of maintaining international peace and security, to develop friendly relations among states, to promote international cooperation, and to serve as a centre for harmonizing the actions of states in achieving those goals. 

  

[Propaganda] Propaganda is a form of communication that is primarily used to influence an audience and persuade them to further an agenda. Propaganda may not be objective and often selectively presents facts to encourage a particular perception, or uses loaded language to produce an emotional rather than a rational response to the information presented. Propaganda can be found in a wide variety of different contexts. 

  

[Globalization] Globalization is the process of increasing interdependence and integration among the economies, markets, societies, and cultures of different countries worldwide. It can be attributed to a series of factors, including the reduction of barriers to international trade, the liberalization of capital movements, the development of transportation infrastructure, and the advancement of information and communication technologies. The term globalization first appeared in the early 20th century, but came into popular use in the 1990s to describe the growing international connectivity of the post–Cold War world. 

[World history (disambiguation)] World history or history of the world usually refers to human history, the history of human beings that takes a human perspective. World history may also refer to: 

History and academics World history (field), or global history, a field of historical study that takes a worldwide/global perspective Big History, an academic discipline that takes an astronomical perspective (from the Big Bang to the present) Chronology of the universe, the history and future of the universe according to Big Bang cosmology History of Earth, the history of planet Earth Recorded history, the history of human beings from when human beings began writing events down (as opposed to prehistory and protohistory) History of globalization, aspect of human history from the perspective of globalization 

Arts and entertainment 

Music World History (album), a 1998 album by Christian rock band Mad at the World "The History of the World (Part 1)", a 1980 song by The Damned 

Film and television History of the World, Part I, a 1981 film by Mel Brooks History of the World, Part II, a 2023 TV series Andrew Marr's History of the World, a 2012 BBC documentary television series presented by Andrew Marr 

Literature History of the World (book), a 1944 book edited by William Nassau Weech The History of the World (Raleigh), a 1614 book 

Gaming History of the World (board game), a 1991 board game designed by Gary Dicken and Steve Kendall History of the World (video game), a 1997 computer game adaptation of the board game 

See also All pages with titles containing World history All pages with titles containing History of the world Timelines of world history Universal history (disambiguation) Universal history (genre), a literary genre History of the Entire World, I Guess, a 2017 video by Bill Wurtz 

[World War II] World War II, or the Second World War (1 September 1939 – 2 September 1945), was a global conflict between two coalitions: the Allies and the Axis powers. Nearly all of the world's countries participated, with many engaging in total war on an unprecedented scale. World War II was the deadliest conflict in history, causing the deaths of 60 to 75 million people, a majority of whom were civilians. Millions died as a result of massacres, starvation, disease, and genocides including the Holocaust. After the Allied victory, Germany, Austria, Japan, and Korea were occupied, and German and Japanese leaders were tried for war crimes. The causes of World War II included unresolved tensions in the aftermath of World War I and the rise of fascism in Europe and militarism in Japan. Key events preceding the war included Japan's invasion of Manchuria in 1931, the Spanish Civil War, the outbreak of the Second Sino-Japanese War in 1937, and Germany's annexations of Austria and the Sudetenland. World War II is generally considered to have begun on 1 September 1939, when Nazi Germany, under Adolf Hitler, invaded Poland, after which the United Kingdom and France declared war on Germany. Poland was also invaded by the Soviet Union in mid-September, and was partitioned between the two states under the Molotov–Ribbentrop Pact. In 1940, the Soviet Union annexed the Baltic states and parts of Finland and Romania, while Germany conquered Norway, Denmark, Belgium, Luxembourg, and the Netherlands. After the fall of France in June 1940, the war continued, mainly between Germany, now assisted by Fascist Italy, and the British Empire and British Commonwealth, with fighting in the Balkans, Mediterranean, Middle East, East Africa, the aerial Battle of Britain, the Blitz, and the naval Battle of the Atlantic. By mid-1941, Yugoslavia and Greece had also been defeated by Axis countries. In June 1941, Germany invaded the Soviet Union, opening the Eastern Front. In December 1941, Japan attacked American and British territories in Asia and the Pacific, including Pearl Harbor in Hawaii, leading the United States to enter the war against the Axis. Japan conquered much of coastal China and Southeast Asia, but its advances in the Pacific were halted in June 1942 at the Battle of Midway. In early 1943, Axis forces were defeated in North Africa and at Stalingrad in the Soviet Union. An Allied invasion of Italy in July resulted in the fall of its fascist regime, and Allied offensives in the Pacific and the Soviet Union forced the Axis to retreat on all fronts. In 1944, the Western Allies invaded France at Normandy, opening a new front, and the Soviet Union advanced into Central Europe. Japan also suffered major setbacks, including the crippling of its navy by the United States, the loss of key Western Pacific islands, and defeats in Burma. The war in Europe concluded with the liberation of German-occupied territories and an invasion of Germany itself, which culminated in the fall of Berlin and Germany's unconditional surrender on 8 May 1945. On 6 and 9 August, the US dropped atomic bombs on Hiroshima and Nagasaki, followed by a Soviet invasion of Japanese-occupied Manchuria. Japan announced its unconditional surrender on 15 August and signed a surrender document on 2 September 1945. World War II transformed the political, economic, and social structures of the world and established the foundation of international relations for the rest of the 20th century and into the 21st century. The United Nations was created to foster international cooperation and prevent future conflicts, with the victorious great powers—China, France, the Soviet Union, the United Kingdom, and the United States—becoming the permanent members of its Security Council. The Soviet Union and the United States emerged as rival superpowers, setting the stage for the Cold War. In the wake of Europe's devastation, the influence of its great powers waned, triggering the decolonisation of Africa and of Asia. Many countries whose industries had been damaged moved towards economic recovery and expansion. 

Start and end dates 

Most historians agree that World War II began with the German invasion of Poland on 1 September 1939 and the British and French declarations of war on Germany two days later. Dates for the beginning of the Pacific War include the start of the Second Sino-Japanese War on 7 July 1937, or the earlier Japanese invasion of Manchuria on 18 September 1931. Other proposed starting dates include the Italian invasion of Abyssinia on 3 October 1935. The British historian Antony Beevor views the beginning of World War II as the Battles of Khalkhin Gol fought between Japan and the forces of Mongolia and the Soviet Union from May to September 1939. Others view the Spanish Civil War as the start of or prelude to World War II. The date of the war's end is also not universally agreed upon. It was generally accepted at the time that the war ended with the armistice of 15 August 1945 (V-J Day), rather than with the formal surrender of Japan on 2 September 1945, which officially ended the war in Asia. A peace treaty between Japan and the Allies was signed in 1951. A 1990 treaty regarding Germany's future allowed the reunification of East and West Germany to take place. No formal peace treaty between Japan and the Soviet Union was ever signed, although the state of war between the two countries was terminated by the Soviet–Japanese Joint Declaration of 1956, which also restored full diplomatic relations between them. 

Background 

Aftermath of World War I 

World War I radically altered the European political map. The most prominent nations of the Central Powers each lost territory in their respective peace treaties at the conclusion of the conflict. New nation-states were created out of the dissolution of the Austro-Hungarian, Ottoman, and Russian Empires. To prevent a future world war, the League of Nations was established in 1920 by the Paris Peace Conference. The organisation's primary function was to prevent armed conflict through collective security, military, and naval disarmament, as well as settling international disputes through peaceful negotiations and arbitration. Despite strong pacifist sentiment after World War I, irredentist and revanchist nationalism had emerged in several European states. These sentiments were especially pronounced in Germany due to the significant territorial, colonial, and financial losses imposed by the Treaty of Versailles. Under the treaty, Germany lost around 13 per cent of its home territory and all its overseas possessions, while German annexation of other states was prohibited, reparations were imposed, and limits were placed on the size and capability of the country's armed forces. 

Germany and Italy The German Empire was dissolved in the German revolution of 1918–1919, and a democratic government, later known as the Weimar Republic, was created. The interwar period saw strife between supporters of the new republic and hardline opponents on both the political right and left. Italy, as an Entente ally, had made some post-war territorial gains; however, Italian nationalists were angered that the promises made by the United Kingdom and France to secure Italian entrance into the war were not fulfilled in the peace settlement. From 1922 to 1925, the fascist movement led by Benito Mussolini seized power in Italy with a nationalist, totalitarian, and class collaborationist agenda that abolished representative democracy, repressed socialist, left-wing, and liberal forces and pursued an aggressive expansionist foreign policy aimed at making Italy a world power, promising the creation of a "New Roman Empire". 

Adolf Hitler, after an unsuccessful attempt to overthrow the German government in 1923, eventually became the chancellor of Germany in 1933 when President Paul von Hindenburg and the Reichstag appointed him. The Nazis soon abolished parliamentary democracy, espousing a radical, racially motivated revision of the world order, and began a massive rearmament campaign. Following Hindenburg's death in 1934, Hitler proclaimed himself Führer of Germany. France, seeking to secure its alliance with Italy, allowed Italy a free hand in Ethiopia, which Italy desired as a colonial possession. The situation was aggravated in early 1935 when the Territory of the Saar Basin was legally reunited with Germany, and Hitler repudiated the Treaty of Versailles, accelerated his rearmament programme, and introduced conscription. 

European treaties The United Kingdom, France and Italy formed the Stresa Front in April 1935 to contain Germany, a key step towards military globalisation; however, that June, the United Kingdom made an independent naval agreement with Germany, easing prior restrictions. The Soviet Union, concerned by Germany's goals of capturing vast areas of Eastern Europe, drafted a treaty of mutual assistance with France. Before taking effect, though, the Franco-Soviet pact was required to go through the bureaucracy of the League of Nations, which rendered it essentially toothless. The United States, concerned with events in Europe and Asia, passed the Neutrality Act in August of the same year. Hitler defied the Versailles and Locarno Treaties by remilitarising the Rhineland in March 1936, encountering little opposition due to the policy of appeasement. In October 1936, Germany and Italy formed the Rome–Berlin Axis. A month later, Germany and Japan signed the Anti-Comintern Pact, which Italy joined the following year. 

Asia The Kuomintang party in China launched a unification campaign against regional warlords and nominally unified China in the mid-1920s, but was soon embroiled in a civil war against its former Chinese Communist Party (CCP) allies and new regional warlords. In 1931, an increasingly militaristic Empire of Japan, which had long sought influence in China as the first step of what its government saw as the country's right to rule Asia, staged the Mukden incident as a pretext to invade Manchuria and establish the puppet state of Manchukuo. China appealed to the League of Nations to stop the Japanese invasion of Manchuria. Japan withdrew from the League of Nations after being condemned for its incursion into Manchuria. The two nations then fought several battles, in Shanghai, Rehe, and Hebei, until the Tanggu Truce was signed in 1933. Thereafter, Chinese volunteer forces continued the resistance to Japanese aggression in Manchuria, and Chahar and Suiyuan. After the 1936 Xi'an Incident, the Kuomintang and CCP forces agreed on a ceasefire to present a united front to oppose Japan. 

Pre-war events 

Italian invasion of Ethiopia (1935) 

The Second Italo-Ethiopian War was a colonial war that began in October 1935 and ended in May 1936. The war began with the invasion of the Ethiopian Empire (also known as Abyssinia) by the armed forces of the Kingdom of Italy (Regno d'Italia), which was launched from Italian Somaliland and Eritrea. The war resulted in the military occupation of Ethiopia and its annexation into the newly created colony of Italian East Africa (Africa Orientale Italiana); in addition it exposed the weakness of the League of Nations as a force to preserve peace. Both Italy and Ethiopia were member nations, but the League did little when the former clearly violated Article X of the League's Covenant. The United Kingdom and France supported imposing sanctions on Italy for the invasion, although the sanctions were not fully enforced and failed to end the Italian invasion. Italy subsequently dropped its objections to Germany's goal of absorbing Austria. 

Spanish Civil War (1936–1939) 

When civil war broke out in Spain, Hitler and Mussolini lent military support to the Nationalist rebels, led by General Francisco Franco. Italy supported the Nationalists to a greater extent than the Nazis: Mussolini sent more than 70,000 ground troops, 6,000 aviation personnel, and 720 aircraft to Spain. The Soviet Union supported the existing government of the Spanish Republic. More than 30,000 foreign volunteers, known as the International Brigades, also fought against the Nationalists. Both Germany and the Soviet Union used this proxy war as an opportunity to test in combat their most advanced weapons and tactics. The Nationalists won the civil war in April 1939; Franco, now dictator, remained officially neutral during World War II but generally favoured the Axis. His largest collaboration with Germany was the sending of volunteers to fight on the Eastern Front. 

Japanese invasion of China (1937) 

In July 1937, Japan captured the former Chinese imperial capital of Peking after instigating the Marco Polo Bridge incident, which culminated in the Japanese campaign to invade all of China following years of tension and low-level conflicts. The Soviets quickly signed a non-aggression pact with China to lend materiel support, effectively ending China's prior cooperation with Germany. From September to November, the Japanese attacked Taiyuan, engaged the Kuomintang Army around Xinkou, fought communist forces in Pingxingguan and wrestled control over China's northern railway network. Nationalist Generalissimo Chiang Kai-shek deployed his best army to defend Shanghai, but after three months of heavy fighting, Shanghai fell. The Japanese continued to push Chinese forces back, capturing the capital Nanking in December 1937. In March 1938, Nationalist Chinese forces won their first major victory at Taierzhuang, but ultimately lost control of the city of Xuzhou in May. In June 1938, Chinese forces stalled the Japanese advance by flooding the Yellow River; buying time for the Chinese to prepare their defences at Wuhan at heavy cost to the local civilian population, but the city was taken by October after heavy fighting along the Yangtze River. Japanese military victories did not destroy Chinese resistance; instead, the Chinese government relocated inland to Chongqing and continued the war. Aiming to break Chinese morale, Japanese aircraft began striking cities in the Sichuan basin in a bombing campaign, killing tens of thousands of civilians. 

Soviet–Japanese border conflicts 

In the mid-to-late 1930s, Japanese forces in Manchukuo had sporadic border clashes with the Soviet Union and Mongolia. The Japanese doctrine of Hokushin-ron, which emphasised Japan's expansion northward, was favoured by the Imperial Army during this time. This policy would prove difficult to maintain in light of the Japanese defeat at Khalkin Gol in 1939, the ongoing Second Sino-Japanese War and ally Nazi Germany pursuing neutrality with the Soviets. Japan and the Soviet Union eventually signed a Neutrality Pact in April 1941, and Japan adopted the doctrine of Nanshin-ron, promoted by the Navy, which took its focus southward and eventually led to war with the United States and the Western Allies. 

European occupations and agreements 

In Europe, Germany and Italy were becoming more aggressive. In March 1938, Germany annexed Austria, again provoking little response from other European powers. Encouraged, Hitler began pressing German claims on the Sudetenland, an area of Czechoslovakia with a predominantly ethnic German population. Soon the United Kingdom and France followed the appeasement policy of British Prime Minister Neville Chamberlain and conceded this territory to Germany in the Munich Agreement, which was made against the wishes of the Czechoslovak government, in exchange for a promise of no further territorial demands. Soon afterwards, Germany and Italy forced Czechoslovakia to cede additional territory to Hungary, and Poland annexed the Trans-Olza region of Czechoslovakia. Although all of Germany's stated demands had been satisfied by the agreement, privately Hitler was furious that British interference had prevented him from seizing all of Czechoslovakia in one operation. In subsequent speeches Hitler attacked British and Jewish "war-mongers" and in January 1939 secretly ordered a major build-up of the German navy to challenge British naval supremacy. In March 1939, Germany invaded the remainder of Czechoslovakia and subsequently split it into the German Protectorate of Bohemia and Moravia and a pro-German client state, the Slovak Republic. Hitler also delivered an ultimatum to Lithuania on 20 March 1939, forcing the concession of the Klaipėda Region, formerly the German Memelland. Following further demands from Hitler regarding the Free City of Danzig and the Polish corridor, the United Kingdom and France guaranteed their support for Polish independence. When Italy conquered Albania in April 1939, the same guarantee was extended to the Kingdoms of Romania and Greece. Shortly after the Franco-British pledge to Poland, Germany and Italy formalised their own alliance with the Pact of Steel. Hitler accused the United Kingdom and Poland of trying to encircle Germany and renounced the Anglo-German Naval Agreement and the German–Polish declaration of non-aggression. 

The situation became a crisis in late August as Germany concentrated its forces near the Polish border. On 23 August, the Soviet Union signed a non-aggression pact with Germany, after tripartite negotiations for a military alliance between France, the United Kingdom, and Soviet Union had stalled. This pact had a secret protocol that defined German and Soviet spheres of influence: Lithuania for Germany; Finland, Estonia, Latvia, and Bessarabia for the Soviet Union; and Poland to be partitioned between the two powers. The pact ensured that Germany would not face a war with the Soviet Union when it invaded Poland. Immediately afterwards, Hitler ordered the attack to proceed on 26 August, but upon hearing that the United Kingdom had concluded a formal mutual assistance pact with Poland and that Italy would maintain neutrality, he decided to delay it. Germany offered Britain an alliance if Britain helped Germany gain Danzig, the Polish corridor and its former colonies, but Britain replied that it would honour its guarantee to Poland. On 29 August, Hitler demanded that a Polish plenipotentiary travel to Berlin by the following day to negotiate a solution to the crisis, but Britain rejected the timeline as unreasonable. On the night of 30–31 August, the British ambassador Nevile Henderson met German Foreign Minister Joachim von Ribbentrop. Ribbentrop handed him Hitler's demands regarding Poland then announced that the deadline for Poland's acceptance had already passed. 

Course of the war 

War breaks out in Europe (1939–1940) 

On 1 September 1939, Germany invaded Poland after having staged several false flag border incidents as a pretext to initiate the invasion. The first German attack of the war came against the Polish defences at Westerplatte. The United Kingdom responded with an ultimatum for Germany to cease military operations, and on 3 September, after the ultimatum was ignored, Britain and France declared war on Germany. During the Phoney War period, the alliance provided no direct military support to Poland, outside of a cautious French probe into the Saarland. The Western Allies also began a naval blockade of Germany, which aimed to damage the country's economy and war effort. Germany responded by ordering U-boat warfare against Allied merchant and warships, which would later escalate into the Battle of the Atlantic. On 8 September, German troops reached the suburbs of Warsaw. The Polish counter-offensive to the west halted the German advance for several days, but it was outflanked and encircled by the Wehrmacht. Remnants of the Polish army broke through to besieged Warsaw. On 17 September 1939, two days after signing a cease-fire with Japan, the Soviet Union invaded Poland under the supposed pretext that the Polish state had ceased to exist. On 27 September, the Warsaw garrison surrendered to the Germans, and the last large operational unit of the Polish Army surrendered on 6 October. Despite the military defeat, Poland never surrendered; instead, it formed the Polish government-in-exile and a clandestine state apparatus remained in occupied Poland. A significant part of Polish military personnel evacuated to Romania and Latvia; many of them later fought against the Axis in other theatres of the war. Germany annexed western Poland and occupied central Poland; the Soviet Union annexed eastern Poland. Small shares of Polish territory were transferred to Lithuania and Slovakia. On 6 October, Hitler made a public peace overture to the United Kingdom and France but said that the future of Poland was to be determined exclusively by Germany and the Soviet Union. The proposal was rejected and Hitler ordered an immediate offensive against France, which was postponed until the spring of 1940 due to bad weather. 

After the outbreak of war in Poland, Stalin threatened Estonia, Latvia, and Lithuania with military invasion, forcing the three Baltic countries to sign pacts allowing the creation of Soviet military bases in these countries; in October 1939, significant Soviet military contingents were moved there. Finland refused to sign a similar pact and rejected ceding part of its territory to the Soviet Union. The Soviet Union invaded Finland in November 1939, and was subsequently expelled from the League of Nations for this crime of aggression. Despite overwhelming numerical superiority, Soviet military success during the Winter War was modest, and the Finno–Soviet war ended in March 1940 with some Finnish concessions of territory. In June 1940, the Soviet Union occupied the entire territories of Estonia, Latvia, and Lithuania, as well as the Romanian regions of Bessarabia, Northern Bukovina, and the Hertsa region. In August 1940, Hitler imposed the Second Vienna Award on Romania which led to the transfer of Northern Transylvania to Hungary. In September 1940, Bulgaria demanded Southern Dobruja from Romania with German and Italian support, leading to the Treaty of Craiova. The loss of one-third of Romania's 1939 territory caused a coup against King Carol II, turning Romania into a fascist dictatorship under Marshal Ion Antonescu, with a course set towards the Axis in the hopes of a German guarantee. Meanwhile, German–Soviet political relations and economic co-operation gradually stalled, and both states began preparations for war. 

Western Europe (1940–1941) 

In April 1940, Germany invaded Denmark and Norway to protect shipments of iron ore from Sweden, which the Allies were attempting to cut off. Denmark capitulated after six hours, and despite Allied support, Norway was conquered within two months. British discontent over the Norwegian campaign led to the resignation of Prime Minister Neville Chamberlain, who was replaced by Winston Churchill on 10 May 1940. On the same day, Germany launched an offensive against France. To circumvent the strong Maginot Line fortifications on the Franco-German border, Germany directed its attack at the neutral nations of Belgium, the Netherlands, and Luxembourg. The Germans carried out a flanking manoeuvre through the Ardennes region, which was mistakenly perceived by the Allies as an impenetrable natural barrier against armoured vehicles. By successfully implementing new Blitzkrieg tactics, the Wehrmacht rapidly advanced to the Channel and cut off the Allied forces in Belgium, trapping the bulk of the Allied armies in a cauldron on the Franco-Belgian border near Lille. The United Kingdom was able to evacuate a significant number of Allied troops from the continent by early June, although they had to abandon almost all their equipment. On 10 June, Italy invaded France, declaring war on both France and the United Kingdom. The Germans turned south against the weakened French army, and Paris fell to them on 14 June. Eight days later France signed an armistice with Germany; it was divided into German and Italian occupation zones, and an unoccupied rump state under the Vichy Regime, which, though officially neutral, was generally aligned with Germany. France kept its fleet, which the United Kingdom attacked on 3 July in an attempt to prevent its seizure by Germany. 

The air Battle of Britain began in early July with Luftwaffe attacks on shipping and harbours. The German campaign for air superiority started in August but its failure to defeat RAF Fighter Command forced the indefinite postponement of the proposed German invasion of Britain. The German strategic bombing offensive intensified with night attacks on London and other cities in the Blitz, but largely ended in May 1941 after failing to significantly disrupt the British war effort. Using newly captured French ports, the German Navy enjoyed success against an over-extended Royal Navy, using U-boats against British shipping in the Atlantic. The British Home Fleet scored a significant victory on 27 May 1941 by sinking the German battleship Bismarck. In November 1939, the United States was assisting China and the Western Allies, and had amended the Neutrality Act to allow "cash and carry" purchases by the Allies. In 1940, following the German capture of Paris, the size of the United States Navy was significantly increased. In September the United States further agreed to a trade of American destroyers for British bases. Still, a large majority of the American public continued to oppose any direct military intervention in the conflict well into 1941. In December 1940, President Franklin D. Roosevelt accused Hitler of planning world conquest and ruled out any negotiations as useless, calling for the United States to become an "arsenal of democracy" and promoting Lend-Lease programmes of military and humanitarian aid to support the British war effort; Lend-Lease was later extended to the other Allies, including the Soviet Union after it was invaded by Germany. The United States started strategic planning to prepare for a full-scale offensive against Germany. At the end of September 1940, the Tripartite Pact formally united Japan, Italy, and Germany as the Axis powers. The Tripartite Pact stipulated that any country—with the exception of the Soviet Union—that attacked any Axis Power would be forced to go to war against all three. The Axis expanded in November 1940 when Hungary, Slovakia, and Romania joined. Romania and Hungary later made major contributions to the Axis war against the Soviet Union, in Romania's case partially to recapture territory ceded to the Soviet Union. 

Mediterranean (1940–1941) 

In early June 1940, the Italian Regia Aeronautica attacked and besieged Malta, a British possession. From late summer to early autumn, Italy conquered British Somaliland and made an incursion into British-held Egypt. In October, Italy attacked Greece, but the attack was repulsed with heavy Italian casualties; the campaign ended within months with minor territorial changes. To assist Italy and prevent Britain from gaining a foothold, Germany prepared to invade the Balkans, which would threaten Romanian oil fields and strike against British dominance of the Mediterranean. 

In December 1940, British Empire forces began counter-offensives against Italian forces in Egypt and Italian East Africa. The offensives were successful; by early February 1941, Italy had lost control of eastern Libya, and large numbers of Italian troops had been taken prisoner. The Italian Navy also suffered significant defeats, with the Royal Navy putting three Italian battleships out of commission after a carrier attack at Taranto, and neutralising several more warships at the Battle of Cape Matapan. Italian defeats prompted Germany to deploy an expeditionary force to North Africa; at the end of March 1941, Erwin Rommel's Afrika Korps launched an offensive which drove back Commonwealth forces. In less than a month, Axis forces advanced to western Egypt and besieged the port of Tobruk. In November Commonwealth forces launched a counter-offensive in North Africa and reclaimed all the gains the Germans and Italians had made. By late March 1941, Bulgaria and Yugoslavia signed the Tripartite Pact; however, the Yugoslav government was overthrown two days later by pro-British nationalists. Germany and Italy responded with simultaneous invasions of both Yugoslavia and Greece, commencing on 6 April 1941 with a massive bombing of Belgrade; both nations were forced to surrender within the month. The airborne invasion of the Greek island of Crete at the end of May completed the German conquest of the Balkans. Armed resistance soon emerged in Yugoslavia and Greece and continued until the end of the war. In the Middle East in May, Commonwealth forces quashed an uprising in Iraq which had been supported by German aircraft from bases within Vichy-controlled Syria. Between June and July, British-led forces invaded the French possessions of Syria and Lebanon, assisted by the Free French. 

Axis attack on the Soviet Union (1941) 

With the situation in Europe and Asia relatively stable, Germany, Japan, and the Soviet Union made preparations for war. With the Soviets wary of mounting tensions with Germany, and the Japanese planning to take advantage of the European War by seizing resource-rich European possessions in Southeast Asia, the two powers signed the Soviet–Japanese Neutrality Pact in April 1941. By contrast, the Germans were steadily making preparations for an attack on the Soviet Union, massing forces on the Soviet border. Hitler believed that the United Kingdom's refusal to end the war was based on the hope that the United States and the Soviet Union would enter the war against Germany. On 31 July 1940, Hitler decided that the Soviet Union should be eliminated and aimed for the conquest of Ukraine, the Baltic states, and Byelorussia. However, other senior German officials like Ribbentrop saw an opportunity to create a Euro-Asian bloc against the British Empire by inviting the Soviet Union into the Tripartite Pact. In November 1940, negotiations took place to determine if the Soviet Union would join the pact. The Soviets showed some interest but asked for concessions from Finland, Bulgaria, Turkey, and Japan that Germany considered unacceptable. On 18 December 1940, Hitler issued the directive to prepare for an invasion of the Soviet Union. On 22 June 1941, Germany, supported by Italy and Romania, invaded the Soviet Union in Operation Barbarossa, with Germany accusing the Soviets of plotting against them; they were joined shortly by Finland and Hungary. The primary targets of this surprise offensive were the Baltic region, Moscow and Ukraine, with the ultimate goal of ending the 1941 campaign near the Arkhangelsk–Astrakhan line—from the Caspian to the White Seas. Hitler's objectives were to eliminate the Soviet Union as a military power, exterminate communism, generate Lebensraum ("living space") by dispossessing the native population, and guarantee access to the strategic resources needed to defeat Germany's remaining rivals. 

Although the Red Army was preparing for strategic counter-offensives before the war, Operation Barbarossa forced the Soviet supreme command to adopt strategic defence. During the summer, the Axis made significant gains into Soviet territory, inflicting immense losses in both personnel and materiel, mainly in massive encirclements around Minsk, Smolensk, and Uman. Nazi policy entailed that Wehrmacht subject Soviet POWs to murderous treatment, executing all Jewish and Communist POWs immediately per the Commissar Order, and subjecting the remainder to forced marches to open-air concentration camps, where they were to be deliberately starved to death. By the end of the winter of 1941, 2.8 million Soviet POWs had died in German captivity. Some 3.3 million Soviet POWs would die in German captivity by the war's end in total, a nearly 60% mortality rate. By mid-August, however, the German Army High Command decided to suspend the offensive of a considerably depleted Army Group Centre, and to divert the 2nd Panzer Group to reinforce troops advancing towards central Ukraine and Leningrad. The Kiev offensive was overwhelmingly successful, resulting in encirclement and elimination of four Soviet armies, and made possible further advance into Crimea and industrially-developed eastern Ukraine (the First Battle of Kharkov). The diversion of three-quarters of the Axis troops and the majority of their air forces from France and the central Mediterranean to the Eastern Front prompted the United Kingdom to reconsider its grand strategy. In July, the UK and the Soviet Union formed a military alliance against Germany and in August, the United Kingdom and the United States jointly issued the Atlantic Charter, which outlined British and American goals for the post-war world. In late August the British and Soviets invaded neutral Iran to secure the Persian Corridor, Iran's oil fields, and preempt any Axis advances through Iran toward the Baku oil fields or India. 

By October, Axis powers had achieved operational objectives in Ukraine and the Baltic region, with only the sieges of Leningrad and Sevastopol continuing. A major offensive against Moscow was renewed; after two months of fierce battles in increasingly harsh weather, the German army almost reached the outer suburbs of Moscow, where the exhausted troops were forced to suspend the offensive. Large territorial gains were made by Axis forces, but their campaign had failed to achieve its main objectives: two key cities remained in Soviet hands, the Soviet capability to resist was not broken, and the Soviet Union retained a considerable part of its military potential. The blitzkrieg phase of the war in Europe had ended. By early December, freshly mobilised reserves allowed the Soviets to achieve numerical parity with Axis troops. This, as well as intelligence data which established that a minimal number of Soviet troops in the East would be sufficient to deter any attack by the Japanese Kwantung Army, allowed the Soviets to begin a massive counter-offensive that started on 5 December all along the front and pushed German troops 100–250 kilometres (62–155 mi) west. 

War breaks out in the Pacific (1941) 

Following the Japanese false flag Mukden incident in 1931, the Japanese shelling of the American gunboat USS Panay in 1937, and the 1937–1938 Nanjing massacre, Japanese-American relations deteriorated. In 1939, the United States notified Japan that it would not be extending its trade treaty and American public opinion opposing Japanese expansionism led to a series of economic sanctions—the Export Control Acts—which banned US exports of chemicals, minerals and military parts to Japan, and increased economic pressure on the Japanese regime. During 1939 Japan launched its first attack against Changsha, but was repulsed by late September. Despite several offensives by both sides, by 1940 the war between China and Japan was at a stalemate. To increase pressure on China by blocking supply routes, and to better position Japanese forces in the event of a war with the Western powers, Japan invaded and occupied northern Indochina in September 1940. Chinese nationalist forces launched a large-scale counter-offensive in early 1940. In August, Chinese communists launched an offensive in Central China; in retaliation, Japanese armies in North China implemented the Three Alls policy, a massive scorched earth initiative to depopulate regions deemed hostile to Japanese occupation. Continued antipathy between Chinese communist and nationalist forces culminated in armed clashes in January 1941, effectively ending their co-operation. In March, the Japanese 11th army attacked the headquarters of the nationalist Chinese 19th army but was repulsed during the Battle of Shanggao. In September, Japan attempted to take the city of Changsha again and clashed with Chinese nationalist forces. German successes in Europe prompted Japan to increase pressure on European governments in Southeast Asia. The Dutch government agreed to provide Japan with oil supplies from the Dutch East Indies, but negotiations for additional access to their resources ended in failure in June 1941. In July 1941 Japan sent troops to southern Indochina, threatening British and Dutch possessions in the Far East. The United States, the United Kingdom, and other Western governments reacted to this move with a freeze on Japanese assets and a total oil embargo. At the same time, Japan was planning an invasion of the Soviet Far East, intending to take advantage of the German invasion in the west, but abandoned the operation after the sanctions. Since early 1941, the United States and Japan had been engaged in negotiations in an attempt to improve their strained relations and end the war in China. Japan advanced a number of proposals which were dismissed by the Americans as inadequate. At the same time the United States, the United Kingdom, and the Netherlands engaged in secret discussions for the joint defence of their territories, in the event of a Japanese attack against any of them. Roosevelt reinforced the Philippines (an American protectorate scheduled for independence in 1946) and warned Japan that the United States would react to Japanese attacks against any "neighboring countries". Frustrated at the lack of progress and pressured by American–British–Dutch sanctions, especially in oil, Japan prepared for war. Emperor Hirohito, after initial hesitation about Japan's chances of victory, began to favour Japan's entry into the war. As a result, Prime Minister Fumimaro Konoe resigned. Hirohito refused the recommendation to appoint Prince Naruhiko Higashikuni in his place, choosing War Minister Hideki Tojo instead. On 3 November, Osami Nagano explained in detail the plan of the attack on Pearl Harbor to the Emperor. On 5 November, Hirohito approved in imperial conference the operations plan for the war. On 20 November, the new government presented an interim proposal as its final offer. It called for the end of American aid to China and for lifting the embargo on the supply of oil and other resources to Japan. In exchange, Japan promised not to launch any attacks in Southeast Asia and to withdraw its forces from southern Indochina. The American counter-proposal of 26 November required that Japan evacuate all of China without conditions and conclude non-aggression pacts with all Pacific powers. That meant Japan was essentially forced to choose between abandoning its ambitions in China, or seizing the natural resources it needed in the Dutch East Indies by force; the Japanese military did not consider the former an option, and many officers considered the oil embargo an unspoken declaration of war. 

Japan planned to seize European colonies in Asia to create a large defensive perimeter stretching into the Central Pacific. The Japanese would then be free to exploit the resources of Southeast Asia while exhausting the over-stretched Allies by fighting a defensive war. To prevent American intervention while securing the perimeter, it was further planned to neutralise the United States Pacific Fleet and the American military presence in the Philippines from the outset. On 7 December 1941 (8 December in Asian time zones), Japan attacked British and American holdings with near-simultaneous offensives against Southeast Asia and the Central Pacific. These included an attack on the American fleets at Pearl Harbor and the Philippines, as well as invasions of Guam, Wake Island, Malaya, Thailand, and Hong Kong. These attacks led the United States, United Kingdom, China, Australia, and several other states to formally declare war on Japan, whereas the Soviet Union, being heavily involved in large-scale hostilities with European Axis countries, maintained its neutrality agreement with Japan. Germany, followed by the other Axis states, declared war on the United States in solidarity with Japan, citing as justification the American attacks on German war vessels that had been ordered by Roosevelt. 

Axis advance stalls (1942–1943) On 1 January 1942, the Allied Big Four—the Soviet Union, China, the United Kingdom, and the United States—and 22 smaller or exiled governments issued the Declaration by United Nations, thereby affirming the Atlantic Charter and agreeing not to sign a separate peace with the Axis powers. During 1942, Allied officials debated on the appropriate grand strategy to pursue. All agreed that defeating Germany was the primary objective. The Americans favoured a straightforward, large-scale attack on Germany through France. The Soviets demanded a second front. The British argued that military operations should target peripheral areas to wear out German strength, leading to increasing demoralisation, and bolstering resistance forces; Germany itself would be subject to a heavy bombing campaign. An offensive against Germany would then be launched primarily by Allied armour, without using large-scale armies. Eventually, the British persuaded the Americans that a landing in France was infeasible in 1942 and they should instead focus on driving the Axis out of North Africa. At the Casablanca Conference in early 1943, the Allies reiterated the statements issued in the 1942 Declaration and demanded the unconditional surrender of their enemies. The British and Americans agreed to continue to press the initiative in the Mediterranean by invading Sicily to fully secure the Mediterranean supply routes. Although the British argued for further operations in the Balkans to bring Turkey into the war, in May 1943, the Americans extracted a British commitment to limit Allied operations in the Mediterranean to an invasion of the Italian mainland, and to invade France in 1944. 

Pacific (1942–1943) 

Japanese forces achieved naval victories in the South China Sea, Java Sea, and Indian Ocean, and bombed the Allied naval base at Darwin, Australia. On 16 April, 7,000 British soldiers were encircled by the Japanese 33rd Division during the Battle of Yenangyaung in Burma and rescued by the Chinese 38th Division. Despite stubborn resistance by Filipino and US forces, the Philippine Commonwealth was eventually captured in May, forcing its government into exile. Following the capture of Bataan, Japanese armies forced some 75,000 Filipino and American prisoners on a 42km death march, resulting in thousands of deaths. By the end of April, Japan and its ally Thailand had conquered Malaya, the Dutch East Indies, Singapore, Rabaul, and most of Burma, inflicting severe losses on Allied troops and taking a large number of prisoners. Japanese advances were accompanied by numerous atrocities, including the Sook Ching massacre in Singapore. The only Allied success against Japan was a Chinese victory at Changsha. The Japanese victories left it overconfident and overextended. 

In early May 1942, Japan initiated operations to capture Port Moresby by amphibious assault and thus sever communications and supply lines between the United States and Australia. The planned invasion was thwarted when an Allied task force, centred on two American fleet carriers, fought Japanese naval forces to a draw in the Battle of the Coral Sea. Japan's next plan, motivated by the earlier Doolittle Raid, was to seize Midway Atoll and lure American carriers into battle to be eliminated; as a diversion, Japan would also send forces to occupy the Aleutian Islands in Alaska. In mid-May, Japan started the Zhejiang-Jiangxi campaign in China, with the goal of inflicting retribution on the Chinese who aided the surviving American airmen in the Doolittle Raid by destroying Chinese air bases and fighting against the Chinese 23rd and 32nd Army Groups. In early June, Japan put its operations into action, but the Americans had broken Japanese naval codes in late May and were fully aware of the plans and order of battle, and used this knowledge to achieve a decisive victory at Midway over the Imperial Japanese Navy. With its capacity for aggressive action greatly diminished as a result of the Midway battle, Japan attempted to capture Port Moresby by an overland campaign in the Territory of Papua. The Americans planned a counterattack against Japanese positions in the southern Solomon Islands, primarily Guadalcanal, as a first step towards capturing Rabaul, the main Japanese base in the South Pacific. Both plans started in July, but by mid-September, the Battle for Guadalcanal took priority for the Japanese, and troops in New Guinea were ordered to withdraw from the Port Moresby area to the northern part of the island, where they faced Australian and United States troops in the Battle of Buna–Gona. Guadalcanal soon became a focal point for both sides with heavy commitments of troops and ships in the battle for Guadalcanal, with Japanese forces suffering massive losses in the attrition, especially amongst their elite pilots. By the start of 1943, the Japanese were defeated on the island and withdrew their troops. In Burma, Commonwealth forces mounted two operations. The first was a disastrous offensive into the Arakan region in late 1942 that forced a retreat back to India by May 1943. The second was the insertion of irregular forces behind Japanese frontlines in February which, by the end of April, had achieved mixed results. 

Eastern Front (1942–1943) Despite considerable losses, in early 1942 Germany and its allies stopped a major Soviet offensive in central and southern Russia, keeping most territorial gains they had achieved during the previous year. In May, the Germans defeated Soviet offensives in the Kerch Peninsula and at Kharkov. The fortress city of Sevastopol, which the Red Army had held out against Axis siege for nearly 250 days, was finally seized with the use of massive artillery bombardments and poison gas. 

In June 1942 Germany launched its main summer offensive against southern Russia, to seize the oil fields of the Caucasus and occupy the Kuban steppe, while maintaining positions on the northern and central areas of the front. The Germans split Army Group South into two groups: Army Group A advanced to the lower Don River and struck south-east to the Caucasus, while Army Group B headed towards the Volga River. The Soviet Union decided to make its stand at Stalingrad on the Volga. By mid-November, the Germans had nearly taken Stalingrad in bitter street fighting. The Soviet Union began its second winter counter-offensive, starting with an encirclement of the German Sixth Army at Stalingrad, and an assault on the Rzhev salient near Moscow, though the latter failed. By early February 1943, the German army had taken tremendous losses; German troops at Stalingrad had been defeated, and the front-line had been pushed back beyond its position before the summer offensive. In mid-February, after the Soviet push had tapered off, the Germans launched another attack on Kharkov, creating a salient in their front line around the Soviet city of Kursk. 

Western Europe/Atlantic and Mediterranean (1942–1943) 

Exploiting poor American naval command decisions, the German navy ravaged Allied shipping off the American Atlantic coast. The Germans launched a North African offensive in January 1942, pushing the British back to positions at the Gazala line by early February, followed by a temporary lull in combat which Germany used to prepare for their upcoming offensives. Concerns that the Japanese might use bases in Vichy-held Madagascar caused the British to invade the island in early May 1942. An Axis offensive in Libya forced an Allied retreat deep inside Egypt until Axis forces were stopped at El Alamein. On the Continent, raids of Allied commandos on strategic targets, culminating in the failed Dieppe Raid, demonstrated the Western Allies' inability to launch an invasion of continental Europe without much better preparation, equipment, and operational security. In August 1942, the Allies succeeded in repelling a second attack against El Alamein and, at a high cost, managed to deliver desperately needed supplies to the besieged Malta. A few months later, the Allies commenced an attack of their own in Egypt, dislodging the Axis forces and beginning a drive west across Libya. This attack was followed up shortly after by Anglo-American landings in French North Africa, which resulted in the region joining the Allies. Hitler responded to the French colony's defection by ordering the occupation of Vichy France; although Vichy forces did not resist this violation of the armistice, they managed to scuttle their fleet to prevent its capture by German forces. Axis forces in Africa withdrew into Tunisia, which was conquered by the Allies in May 1943. German operations in the Atlantic also suffered. By May 1943, as Allied counter-measures became increasingly effective, the resulting sizeable German submarine losses forced a temporary halt of the German Atlantic naval campaign. In June 1943, the British and Americans began a strategic bombing campaign against Germany with a goal to disrupt the war economy, reduce morale, and "de-house" the civilian population. The firebombing of Hamburg was among the first attacks in this campaign, inflicting significant casualties and considerable losses on infrastructure of this important industrial centre. 

Allies gain momentum (1943–1944) 

After the Guadalcanal campaign, the Allies initiated several operations against Japan in the Pacific. In May 1943, Canadian and US forces were sent to eliminate Japanese forces from the Aleutians. Soon after, the United States, with support from Australia, New Zealand, and Pacific Islander forces, began major ground, sea and air operations to isolate Rabaul by capturing surrounding islands, and breach the Japanese Central Pacific perimeter at the Gilbert and Marshall Islands. By the end of March 1944, the Allies had completed both of these objectives and had also neutralised the major Japanese base at Truk in the Caroline Islands. In April, the Allies launched an operation to retake Western New Guinea. In the Soviet Union, both the Germans and the Soviets spent the spring and early summer of 1943 preparing for large offensives in central Russia. On 5 July 1943, Germany attacked Soviet forces around the Kursk Bulge. Within a week, German forces had exhausted themselves against the Soviets' well-constructed defences, and for the first time in the war, Hitler cancelled an operation before it had achieved tactical or operational success. This decision was partially affected by the Western Allies' invasion of Sicily launched on 9 July, which, combined with previous Italian failures, resulted in the ousting and arrest of Mussolini later that month. 

On 12 July 1943, the Soviets launched their own counter-offensives, thereby nearly completely dispelling any chance of German victory or even stalemate in the east. The Soviet victory at Kursk marked the end of German superiority, giving the Soviet Union the initiative on the Eastern Front. The Germans tried to stabilise their eastern front along the hastily fortified Panther–Wotan line, but the Soviets broke through it at Smolensk and the Lower Dnieper Offensive. On 3 September 1943, the Western Allies invaded the Italian mainland, following Italy's armistice with the Allies and the ensuing German occupation of Italy. Germany, with the help of local fascists, responded to the armistice by disarming Italian forces that were in many places without superior orders, seizing military control of Italian areas, and creating a series of defensive lines. German special forces then rescued Mussolini, who then soon established a new client state in German-occupied Italy named the Italian Social Republic, causing an Italian civil war. The Western Allies fought through several lines until reaching the main German defensive line in mid-November. In November 1943, Franklin D. Roosevelt and Winston Churchill met with Chiang Kai-shek in Cairo and then with Joseph Stalin in Tehran. The former conference determined the post-war return of Japanese territory and the military planning for the Burma campaign, while the latter included agreement that the Western Allies would invade Europe in 1944 and that the Soviet Union would declare war on Japan within three months of Germany's defeat. From November 1943, during the seven-week Battle of Changde, the Chinese awaited Allied relief as they forced Japan to fight a costly war of attrition. In January 1944, the Allies launched a series of attacks in Italy against the line at Monte Cassino and tried to outflank it with landings at Anzio. On 27 January 1944, Soviet troops launched a major offensive that expelled German forces from the Leningrad region, thereby ending the most lethal siege in history. The following Soviet offensive was halted on the pre-war Estonian border by the German Army Group North aided by Estonians hoping to re-establish national independence. This delay slowed subsequent Soviet operations in the Baltic Sea region. By late May 1944, the Soviets had liberated Crimea, largely expelled Axis forces from Ukraine, and made incursions into Romania, which were repulsed by the Axis troops. The Allied offensives in Italy had succeeded and, at the cost of allowing several German divisions to retreat, Rome was captured on 4 June. The Allies had mixed success in mainland Asia. In March 1944, the Japanese launched the first of two invasions, an operation against Allied positions in Assam, India, and soon besieged Commonwealth positions at Imphal and Kohima. In May 1944, British and Indian forces mounted a counter-offensive that drove Japanese troops back to Burma by July, and Chinese forces that had invaded northern Burma in late 1943 besieged Japanese troops in Myitkyina. The second Japanese invasion of China aimed to destroy China's main fighting forces, secure railways between Japanese-held territory and capture Allied airfields. By June, the Japanese had conquered the province of Henan and begun a new attack on Changsha. 

Allied offensives (1944) 

On 6 June 1944 (commonly known as D-Day), after three years of Soviet pressure, the Western Allies invaded northern France. After reassigning several Allied divisions from Italy, they also attacked southern France. These landings were successful and led to the defeat of the German Army units in France. The liberation of Paris on 25 August by the local resistance was assisted by the Free French Forces, both led by General Charles de Gaulle, and the Western Allies continued to push back German forces in western Europe during the latter part of the year. An attempt to advance into northern Germany spearheaded by a major airborne operation in the Netherlands failed. After that, the Western Allies slowly pushed into Germany, but failed to cross the Roer river. In Italy, the Allied advance slowed due to the last major German defensive line. On 22 June, the Soviets launched a strategic offensive in Belarus that nearly destroyed the German Army Group Centre. Soon after that, another Soviet strategic offensive forced German troops from Western Ukraine and Eastern Poland. The Soviet Red Army however halted in the Praga district on the other side of the Vistula as the Germans quelled the Warsaw Uprising initiated by the Home Army (the main faction of the Polish resistance, loyal to the non-communist government-in exile), killing over 150,000 Poles. The national uprising in Slovakia was also quelled by the Germans. The Soviet Red Army's strategic offensive in eastern Romania cut off and destroyed the considerable German troops there and triggered a successful coup d'état in Romania and in Bulgaria, followed by those countries' shift to the Allied side. 

In September 1944, Soviet troops advanced into Yugoslavia and forced the rapid withdrawal of German Army Groups E and F in Greece, Albania, and Yugoslavia to rescue them from being cut off. By this point, the communist-led Partisans under Marshal Josip Broz Tito, who had led an increasingly successful guerrilla campaign against the occupation since 1941, controlled much of the territory of Yugoslavia and engaged in delaying efforts against German forces further south. In northern Serbia, the Soviet Red Army, with limited support from Bulgarian forces, assisted the Partisans in a joint liberation of the capital city of Belgrade on 20 October. A few days later, the Soviets launched a massive assault against German-occupied Hungary that lasted until the fall of Budapest in February 1945. Unlike rapid Soviet victories in the Balkans, bitter Finnish resistance to the Soviet offensive in the Karelian Isthmus denied the Soviets occupation of Finland and led to a Soviet-Finnish armistice on relatively mild conditions, although Finland was obligated to fight their German former allies. By the start of July 1944, Commonwealth forces in Southeast Asia had repelled the Japanese sieges in Assam, pushing the Japanese back to the Chindwin River while the Chinese captured Myitkyina. In September 1944, Chinese forces captured Mount Song and reopened the Burma Road. In China, the Japanese had more successes, having finally captured Changsha in mid-June and the city of Hengyang by early August. Soon after, they invaded the province of Guangxi, winning major engagements against Chinese forces at Guilin and Liuzhou by the end of November and successfully linking up their forces in China and Indochina by mid-December. In the Pacific, US forces continued to push back the Japanese perimeter. In mid-June 1944, they began their offensive against the Mariana and Palau islands and decisively defeated Japanese forces in the Battle of the Philippine Sea. These defeats led to the resignation of the Japanese Prime Minister, Hideki Tojo, and provided the United States with air bases to launch intensive heavy bomber attacks on the Japanese home islands. In late October, American forces invaded the Filipino island of Leyte; soon after, Allied naval forces scored another large victory in the Battle of Leyte Gulf, one of the largest naval battles in history. 

Axis collapse and Allied victory (1944–1945) 

On 16 December 1944, Germany made a last attempt to split the Allies on the Western Front by using most of its remaining reserves to launch a massive counter-offensive in the Ardennes and along the French-German border, hoping to encircle large portions of Western Allied troops and prompt a political settlement after capturing their primary supply port at Antwerp. By 16 January 1945, this offensive had been repulsed with no strategic objectives fulfilled. In Italy, the Western Allies remained stalemated at the German defensive line. In mid-January 1945, the Red Army attacked in Poland, pushing from the Vistula to the Oder river in Germany, and overran East Prussia. On 4 February, Soviet, British, and US leaders met for the Yalta Conference. They agreed on the occupation of post-war Germany, and on when the Soviet Union would join the war against Japan. In February, the Soviets entered Silesia and Pomerania, while the Western Allies entered western Germany and closed to the Rhine River. By March, the Western Allies crossed the Rhine north and south of the Ruhr, encircling the German Army Group B. In early March, in an attempt to protect its last oil reserves in Hungary and retake Budapest, Germany launched its last major offensive against Soviet troops near Lake Balaton. Within two weeks, the offensive had been repulsed, the Soviets advanced to Vienna, and captured the city. In early April, Soviet troops captured Königsberg, while the Western Allies finally pushed forward in Italy and swept across western Germany capturing Hamburg and Nuremberg. American and Soviet forces met at the Elbe river on 25 April, leaving unoccupied pockets in southern Germany and around Berlin. Soviet troops then stormed and captured Berlin in late April. In Italy, German forces surrendered on 29 April, while the Italian Social Republic capitulated two days later. On 30 April, the Reichstag was captured, signalling the military defeat of Nazi Germany. Major changes in leadership occurred on both sides during this period. On 12 April, President Roosevelt died and was succeeded by his vice-president, Harry S. Truman. Benito Mussolini was killed by Italian partisans on 28 April. On 30 April, Hitler committed suicide in his headquarters, and was succeeded by Grand Admiral Karl Dönitz (as President of the Reich) and Joseph Goebbels (as Chancellor of the Reich). Goebbels also committed suicide on the following day and was replaced by Lutz Graf Schwerin von Krosigk, in what would later be known as the Flensburg Government. Total and unconditional surrender in Europe was signed on 7 and 8 May, to be effective by the end of 8 May. German Army Group Centre resisted in Prague until 11 May. On 23 May, all remaining members of the German government were arrested by Allied forces in Flensburg. On 5 June, all German political and military institutions were placed under Allied control through the Berlin Declaration. 

In the Pacific theatre, American forces accompanied by the forces of the Philippine Commonwealth advanced in the Philippines, clearing Leyte by the end of April 1945. They landed on Luzon in January 1945 and recaptured Manila in March, during which Japanese forces killed 100,000 Filipino civilians in the city. Fighting continued on Luzon, Mindanao, and other islands of the Philippines until the end of the war. Meanwhile, the United States Army Air Forces launched a massive firebombing campaign of strategic cities in Japan in an effort to destroy Japanese war industry and civilian morale. A devastating bombing raid on Tokyo of 9–10 March was the deadliest conventional bombing raid in history. In May 1945, Australian troops landed in Borneo, overrunning the oilfields there. British, American, and Chinese forces defeated the Japanese in northern Burma in March, and the British pushed on to reach Rangoon by 3 May. Chinese forces started a counterattack in the Battle of West Hunan that occurred between 6 April and 7 June 1945. American naval and amphibious forces also moved towards Japan, taking Iwo Jima by March, and Okinawa by the end of June. At the same time, a naval blockade by submarines was strangling Japan's economy and drastically reducing its ability to supply overseas forces. On 11 July, Allied leaders met in Potsdam, Germany. They confirmed earlier agreements about Germany, and the American, British, and Chinese governments reiterated the demand for unconditional surrender of Japan, specifically stating that "the alternative for Japan is prompt and utter destruction". During this conference, the United Kingdom held its general election, and Clement Attlee replaced Churchill as Prime Minister. 

The call for unconditional surrender was rejected by the Japanese government, which believed it would be capable of negotiating for more favourable surrender terms. In early August, the United States dropped atomic bombs on the Japanese cities of Hiroshima and Nagasaki. Between the two bombings, the Soviets, pursuant to the Yalta agreement, declared war on Japan, invaded Japanese-held Manchuria and quickly defeated the Kwantung Army, which was the largest Japanese fighting force. These two events persuaded previously adamant Imperial Army leaders to accept surrender terms. The Red Army also captured the southern part of Sakhalin Island and the Kuril Islands. On the night of 9–10 August 1945, Emperor Hirohito ordered the Japanese cabinet to accept the terms demanded by the Allies in the Potsdam Declaration. On 15 August, the Emperor communicated this decision to the Japanese people through a speech broadcast on the radio (Gyokuon-hōsō, literally "broadcast in the Emperor's voice"). On 15 August 1945, Japan surrendered, with the surrender documents finally signed at Tokyo Bay on the deck of the American battleship USS Missouri on 2 September 1945, ending the war. 

Aftermath 

The Allies established occupation administrations in Austria and Germany, both of which were initially divided between western and eastern occupation zones controlled by the Western Allies and the Soviet Union, respectively. However, their paths soon diverged. In Germany, the western and eastern occupation zones officially ended in 1949, with the respective zones becoming separate countries, West Germany and East Germany. In Austria, however, occupation continued until 1955, when a joint settlement between the Western Allies and the Soviet Union permitted the reunification of Austria as a democratic state officially non-aligned with any political bloc (although in practice having better relations with the Western Allies). A denazification program in Germany led to the prosecution of Nazi war criminals in the Nuremberg trials and the removal of ex-Nazis from power, although this policy moved towards amnesty and re-integration of ex-Nazis into West German society. Germany lost a quarter of its pre-war (1937) territory. Among the eastern territories, Silesia, Neumark, and most of Pomerania were taken over by Poland, and East Prussia was divided between Poland and the Soviet Union, followed by the expulsion to Germany of the nine million Germans from these provinces, as well as three million Germans from the Sudetenland in Czechoslovakia. By the 1950s, one-fifth of West Germans were refugees from the east. The Soviet Union also took over the Polish provinces east of the Curzon Line, from which two million Poles were expelled. Northeastern Romania, parts of eastern Finland, and the Baltic states were annexed into the Soviet Union. Italy lost its monarchy, colonial empire, and some European territories. In an effort to maintain world peace, the Allies formed the United Nations, which officially came into existence on 24 October 1945, and adopted the Universal Declaration of Human Rights in 1948 as a common standard for all member nations. The great powers that were the victors of the war—France, China, the United Kingdom, the Soviet Union, and the United States—became the permanent members of the UN's Security Council. The five permanent members remain so to the present, although there have been two seat changes, between the Republic of China and the People's Republic of China in 1971, and between the Soviet Union and its successor state, the Russian Federation, following the dissolution of the Soviet Union in 1991. The alliance between the Western Allies and the Soviet Union had begun to deteriorate even before the war was over. 

Besides Germany, the rest of Europe was also divided into Western and Soviet spheres of influence. Most eastern and central European countries fell into the Soviet sphere, which led to the establishment of communist-led regimes, with full or partial support of the Soviet occupation authorities. As a result, East Germany, Poland, Hungary, Romania, Bulgaria, Czechoslovakia, and Albania became Soviet satellite states. Communist Yugoslavia conducted a fully independent policy, causing tension with the Soviet Union. A communist uprising in Greece was put down with Anglo-American support and the country remained aligned with the West. Post-war division of the world was formalised by two international military alliances, the United States-led NATO and the Soviet-led Warsaw Pact. The long period of political tensions and military competition between them—the Cold War—would be accompanied by an unprecedented arms race and a number of proxy wars throughout the world. In Asia, the United States led the occupation of Japan and administered Japan's former islands in the Western Pacific, while the Soviets annexed South Sakhalin and the Kuril Islands. Korea, formerly under Japanese colonial rule, was divided and occupied by the Soviet Union in the North and the United States in the South between 1945 and 1948. Separate republics emerged on both sides of the 38th parallel in 1948, each claiming to be the legitimate government for all of Korea, which led ultimately to the Korean War. In China, nationalist and communist forces resumed the civil war in June 1946. Communist forces prevailed and established the People's Republic of China on the mainland, while nationalist forces retreated to Taiwan in 1949. In the Middle East, the Arab rejection of the United Nations Partition Plan for Palestine and the creation of Israel marked the escalation of the Arab–Israeli conflict. While European powers attempted to retain some or all of their colonial empires, their losses of prestige and resources during the war rendered this unsuccessful, leading to decolonisation. The global economy suffered heavily from the war, although participating nations were affected differently. The United States emerged much richer than any other nation, leading to a baby boom, and by 1950 its gross domestic product per person was much greater than that of any of the other powers, and it dominated the world economy. The Allied occupational authorities pursued a policy of industrial disarmament in Western Germany from 1945 to 1948. Due to international trade interdependencies, this policy led to an economic stagnation in Europe and delayed European recovery from the war for several years. At the Bretton Woods Conference in July 1944, the Allied nations drew up an economic framework for the post-war world. The agreement created the International Monetary Fund (IMF) and the International Bank for Reconstruction and Development (IBRD), which later became part of the World Bank Group. The Bretton Woods system lasted until 1973. Recovery began with the mid-1948 currency reform in West Germany, and was sped up by the liberalisation of European economic policy that the US Marshall Plan economic aid (1948–1951) both directly and indirectly caused. The post-1948 West German recovery has been called the German economic miracle. Italy also experienced an economic boom and the French economy rebounded. By contrast, the United Kingdom was in a state of economic ruin, and although receiving a quarter of the total Marshall Plan assistance, more than any other European country, it continued in relative economic decline for decades. The Soviet Union, despite enormous human and material losses, also experienced rapid increases in production in the immediate post-war era, having seized and transferred most of Germany's industrial plants and exacted war reparations from its satellite states. Japan recovered much later. China returned to its pre-war industrial production by 1952. 

Impact 

Casualties 

An estimated 60 million to more than 75 million people died in the war including at least 20 million who died from deprivation, famine and disease. Civilian deaths have been estimated to comprise 67% to 80% of all direct and indirect deaths from the war. The Soviet Union had the highest overall death toll (estimated at 20 million to 28 million), followed by China (at least 15 million), Germany (6 million to 8.7 million) and Poland (5 million to 6.5 million). The countries which sustained the most military deaths were the Soviet Union (about 8.7 million), Germany (around 5.3 million), China (2 million to 3 million) and Japan (1.7 million to 2.5 million). Of the 20 million to 25 million military deaths in the war, the majority were of German and Soviet soldiers, including prisoners of war (POWs), on the Eastern Front. The high civilian death toll relative to military deaths was unusual for major wars up to that time. Some 10 million to 15 million people died of starvation and disease in China and the Soviet Union, and 8 million to 10 million in India and under Japanese occupation elsewhere in Asia and the Pacific. Around 15 million civilians died in genocides and other deliberate killings, while millions in total died in concentration camps, aerial bombings and in combat zones. 

Genocide, war crimes, and crimes against humanity 

In trials following the war, representatives of the Axis powers and their accomplices were convicted of numerous war crimes, crimes against humanity, and complicity in genocides. The Nazis killed about 6 million Jews in a racially motivated genocide known as the Holocaust. They also killed millions of Slavs, over 130 thousand Romani, and members of other groups that they considered racially inferior. Almost 300,000 people with mental and physical disabilities were also systematically killed in Germany and its occupied territories. The killing of civilians and POWs through massacres and deliberate starvation was especially common in the Eastern European and Asia-Pacific theatres. German forces on the Eastern Front destroyed or confiscated available food, massacred civilians in reprisals and routinely shot prisoners. The Japanese killed millions of civilians in occupied areas through massacres and scorched earth strategies; for example, in their "kill all, burn all, loot all" policy in China. In the Nanjing massacre, between 40,000 and 200,000 Chinese civilians and POWs were killed. Japan also used biological weapons in China and in early conflicts against the Soviets. The Soviet Union was responsible for imprisoning, deporting and often executing hundreds of thousands of civilians and POWs from occupied or annexed territories. This included the Katyn massacre of 22,000 Polish officers and intellectuals. War crimes were also committed in civil wars and conflicts between resistance groups in occupied territories. In Yugoslavia, numerous war crimes, including the massacre of civilians and prisoners, were committed by Axis forces and the Axis-aligned Croatian Ustaše. The main resistance groups, the Serbian-Nationalist Chetniks and the Communist-led partisans, also committed massacres and persecutions of their enemies. In Poland, about 100,000 Poles were killed by the Ukrainian Insurgent Army in the Volhynia massacres between 1943 and 1945. About 10,000 Ukrainians were killed by Poles in reprisal attacks. In Greece, Axis forces were mainly responsible for civilian deaths through deliberate starvation and reprisal massacres, although the major resistance forces, ELAS and EDES, also committed war crimes. The Soviet Union, Japan and Germany inflicted high death rates on POWs through executions, starvation, forced labour and other mistreatment. The Soviet Union and Japan had not ratified the Geneva Convention on Prisoners of War, and Germany regarded itself as exempt from the convention on the Eastern Front. About a third of those taken prisoner by the Japanese died, as did almost 60 per cent of Soviet prisoners of the Germans. About a third of Germans taken prisoner by the Soviet Union died in captivity. The mortality rate of German and Japanese prisoners of the Western allies was 1 to 2 per cent. 

Germany, Japan and the Soviet Union made extensive use of forced labour of foreign civilians and POWs. In greater Germany, there were 7.6 million foreign workers (including POWs undertaking forced labour) and about 500,000 slave labourers in concentration camps by late 1944. Those civilians and POWs from Soviet-occupied territories who were deported to the Soviet Union were usually imprisoned in the Soviet forced labour camps, known as the gulag. Between 200,000 and one million Soviet POWs and civilians repatriated from German camps were also sent to the gulag as alleged Axis collaborators where many died from malnutrition, the harsh climate and overwork. The Japanese conscripted millions of foreign civilians and POWs to undertake forced labour in Japan and its occupied territories where they suffered harsh treatment and high death rates. Up to 200,000 Korean and Chinese women were forced into sex slavery. While troops of all the major belligerents committed rapes, the rape of civilians was particularly widespread among the German, Japanese and Soviet military. The post-war international military tribunals found the evidence of rape by German forces "overwhelming". Twenty-nine Japanese defendants, mostly generals, were convicted of complicity in mass rape. Soviet soldiers also committed mass rapes in occupied territories, especially Germany. War crimes were committed by the Western Allied powers, but not on the same scale as those of the Axis powers and the Soviet Union. The Western Allies prosecuted a number of war crimes committed by their own forces, but Allied war crimes were not prosecuted by the post-war international military tribunals. There has been continued debate over whether the area bombing of cities in Germany and Japan, and the atomic bombing of Hiroshima and Nagasaki, were war crimes. 

Occupation 

In Europe, occupation came in two forms. In Western, Northern, and Central Europe (France, Norway, Denmark, the Low Countries, and the annexed portions of Czechoslovakia) Germany established economic policies through which it collected roughly 69.5 billion reichsmarks (27.8 billion US dollars) by the end of the war; this figure does not include the plunder of industrial products, military equipment, raw materials and other goods. Thus, the income from occupied nations was over 40 per cent of the income Germany collected from taxation, a figure which increased to nearly 40 per cent of total German income as the war went on. In the East, the intended gains of Lebensraum were never attained as fluctuating front-lines and Soviet scorched earth policies denied resources to the German invaders. Unlike in the West, the Nazi racial policy encouraged extreme brutality against what it considered to be the "inferior people" of Slavic descent; most German advances were thus followed by mass atrocities and war crimes. The Nazis killed an estimated 2.8 million ethnic Poles in addition to Polish-Jewish victims of the Holocaust. Although by 1942 resistance groups formed in most occupied territories, the assessments of the effectiveness of Soviet partisans and French Resistance suggests that they did not significantly hamper German operations until late 1943. 

In Asia, Japan termed nations under its occupation as being part of the Greater East Asia Co-Prosperity Sphere, essentially a Japanese hegemony which it claimed was for purposes of liberating colonised peoples. Although Japanese forces were sometimes welcomed as liberators from European domination, Japanese war crimes frequently turned local public opinion against them. During Japan's initial conquest, it captured 4,000,000 barrels (640,000 m3) of oil (550,000 tonnes) left behind by retreating Allied forces; and by 1943, was able to get production in the Dutch East Indies up to 50 million barrels (7,900,000 m3) of oil (6.8 million tonnes), 76 per cent of its 1940 output rate. 

Home fronts and production 

In the 1930s, Britain and the United States together controlled almost 75% of world mineral output—essential for projecting military power. In Europe, before the outbreak of the war, the Allies had significant advantages in both population and economics. In 1938, the Western Allies (United Kingdom, France, Poland and the British Dominions) had a 30 per cent larger population and a 30 per cent higher gross domestic product than the European Axis powers (Germany and Italy); including colonies, the Allies had more than a 5:1 advantage in population and a nearly 2:1 advantage in GDP. In Asia at the same time, China had roughly six times the population of Japan but only an 89 per cent higher GDP; this reduces to three times the population and only a 38 per cent higher GDP if Japanese colonies are included. The United States produced about two-thirds of all munitions used by the Allies in World War II, including warships, transports, warplanes, artillery, tanks, trucks, and ammunition. Although the Allies' economic and population advantages were largely mitigated during the initial rapid blitzkrieg attacks of Germany and Japan, they became the decisive factor by 1942, after the United States and Soviet Union joined the Allies and the war evolved into one of attrition. While the Allies' ability to out-produce the Axis was partly due to more access to natural resources, other factors, such as Germany and Japan's reluctance to employ women in the labour force, Allied strategic bombing, and Germany's late shift to a war economy contributed significantly. Additionally, neither Germany nor Japan planned to fight a protracted war, and had not equipped themselves to do so. Germany, Japan and the Soviet Union used millions of slave labourers in war-related industries. 

Advances in technology and its application 

Aircraft were used for reconnaissance, as fighters, bombers, and ground-support, and each role developed considerably. Innovations included airlift (the capability to quickly move limited high-priority supplies, equipment, and personnel); and strategic bombing (the bombing of enemy industrial and population centres to destroy the enemy's ability to wage war). Anti-aircraft weaponry also advanced, including defences such as radar and surface-to-air artillery, in particular the introduction of the proximity fuze. The use of the jet aircraft was pioneered and led to jets becoming standard in air forces worldwide. Advances were made in nearly every aspect of naval warfare, most notably with aircraft carriers and submarines. Although aeronautical warfare had relatively little success at the start of the war, actions at Taranto, Pearl Harbor, and the Coral Sea established the carrier as the dominant capital ship (in place of the battleship). In the Atlantic, escort carriers became a vital part of Allied convoys, increasing the effective protection radius and helping to close the Mid-Atlantic gap. Carriers were also more economical than battleships due to the relatively low cost of aircraft and because they are not required to be as heavily armoured. Submarines, which had proved to be an effective weapon during World War I, were expected by all combatants to be important in the second. The British focused development on anti-submarine weaponry and tactics, such as sonar and convoys, while Germany focused on improving its offensive capability, with designs such as the Type VII submarine and wolfpack tactics. Gradually, improving Allied technologies such as the Leigh Light, Hedgehog, Squid, and homing torpedoes proved effective against German submarines. 

Land warfare changed from the static frontlines of trench warfare of World War I, which had relied on improved artillery that outmatched the speed of both infantry and cavalry, to increased mobility and combined arms. The tank, which had been used predominantly for infantry support in the First World War, had evolved into the primary weapon. In the late 1930s, tank design was considerably more advanced than it had been during World War I, and advances continued throughout the war with increases in speed, armour and firepower. At the start of the war, most commanders thought enemy tanks should be met by tanks with superior specifications. This idea was challenged by the poor performance of the relatively light early tank guns against armour, and German doctrine of avoiding tank-versus-tank combat. This, along with Germany's use of combined arms, were among the key elements of their highly successful blitzkrieg tactics across Poland and France. Many means of destroying tanks, including indirect artillery, anti-tank guns (both towed and self-propelled), mines, short-ranged infantry antitank weapons, and other tanks were used. Even with large-scale mechanisation, infantry remained the backbone of all forces, and throughout the war, most infantry were equipped similarly to World War I. The portable machine gun spread, a notable example being the German MG 34, and various submachine guns which were suited to close combat in urban and jungle settings. The assault rifle, a late war development incorporating many features of the rifle and submachine gun, became the standard post-war infantry weapon for most armed forces. Most major belligerents attempted to solve the problems of complexity and security involved in using large codebooks for cryptography by designing ciphering machines, the most well-known being the German Enigma machine. Development of SIGINT (signals intelligence) and cryptanalysis enabled the countering process of decryption. Notable examples were the Allied decryption of Japanese naval codes and British Ultra, a pioneering method for decoding Enigma that benefited from information given to the United Kingdom by the Polish Cipher Bureau, which had been decoding early versions of Enigma before the war. Another component of military intelligence was deception, which the Allies used to great effect in operations such as Mincemeat and Bodyguard. Other technological and engineering feats achieved during, or as a result of, the war include the world's first programmable computers (Z3, Colossus, and ENIAC), guided missiles and modern rockets, the Manhattan Project's development of nuclear weapons, operations research, the development of artificial harbours, and oil pipelines under the English Channel. Although penicillin was discovered before the war, the development of industrial production technology as well as the mass production and use began during the war. 

See also Lists of World War II topics World War III – Hypothetical future global conflict 

Notes 

References 

Sources 

Further reading Buchanan, Andrew (7 February 2023). "Globalizing the Second World War". Past & Present (258): 246–281. doi:10.1093/pastj/gtab042. ISSN 0031-2746. also see online review Archived 4 May 2024 at the Wayback Machine Gerlach, Christian (2024). Conditions of Violence. Walter de Gruyter GmbH & Co KG. ISBN 978-3-1115-6873-7. 

External links 

West Point Maps of the European War. Archived 23 March 2019 at the Wayback Machine. West Point Maps of the Asian-Pacific War. Archived 23 March 2019 at the Wayback Machine. Atlas of the World Battle Fronts (July 1943 – August 1945) 

[United States] The United States of America (USA), also known as the United States (U.S.) or America, is a country primarily located in North America. It is a federal republic consisting of 50 states and a federal capital district, Washington, D.C. The 48 contiguous states border Canada to the north and Mexico to the south, with the semi-exclave of Alaska in the northwest and the archipelago of Hawaii in the Pacific Ocean. The United States also asserts sovereignty over five major island territories and various uninhabited islands in Oceania and the Caribbean. It is a megadiverse country, with the world's fourth-largest land area and third-largest population, exceeding 341 million. Paleo-Indians migrated from North Asia to North America around 15,000 years ago and formed various civilizations. Beginning with the 1607 settlement of Virginia, British colonization established the Thirteen Colonies. The American Enlightenment, spread throughout the colonies in the 18th century, valued republicanism and liberalism. Clashes with the British Crown over taxation without parliamentary representation, among other denied English rights, sparked the American Revolution, which led to the July 2, 1776, Lee Resolution formally declaring independence from Great Britain. Two days later, the Declaration of Independence was adopted. Victory in the 1775–1783 Revolutionary War brought international recognition of U.S. sovereignty. The U.S. expanded westward through purchase, settlement and conquest of European- and Indigenous-controlled territory. As more states joined the Union, a sectional division over slavery led 11 Southern states to secede and form the Confederate States of America, fighting the Union in the Civil War of 1861–1865. With the United States' victory and reunification, slavery was abolished. By 1900, the U.S. emerged as a great power, a status solidified after its involvement in World War I. Following Japan's attack on Pearl Harbor in 1941, it entered World War II on the Allied side. The war's aftermath left the U.S. and the Soviet Union as rival superpowers, vying for geopolitical dominance during the Cold War. The Soviet Union's collapse in 1991 left the U.S. as the world's sole superpower. The U.S. federal government was established as a republic under the United States Constitution, with executive authority vested in a president. The Constitution created a separation of powers among legislative, executive, and judicial branches. The Congress is a bicameral national legislature composed of the House of Representatives (a lower house based on population) and the Senate (an upper house based on equal representation for each state). Federalism grants substantial autonomy to the 50 states. A developed country, the United States ranks high in economic competitiveness, innovation, and higher education. Its economy accounts for over a quarter of nominal global GDP and has been the world's largest since about 1890. The U.S. is the wealthiest country, with the highest disposable household income per capita among OECD members, though its wealth inequality is highly pronounced. American culture, shaped by centuries of immigration, is diverse and globally influential. The U.S. makes up nearly a third of global military spending and is widely considered to have the most powerful armed forces in the world. A member of numerous international organizations, it plays a major role in global political, cultural, economic, and military affairs. 

Etymology 

Documented use of the phrase "United States of America" dates back to January 2, 1776. On that day, Stephen Moylan, a Continental Army aide to General George Washington, wrote a letter to Joseph Reed, Washington's aide-de-camp, seeking to go "with full and ample powers from the United States of America to Spain" to seek assistance in the Revolutionary War effort. The first known public usage is an anonymous essay published in the Williamsburg newspaper The Virginia Gazette on April 6, 1776. Sometime on or after June 11, 1776, Thomas Jefferson wrote "United States of America" in a rough draft of the Declaration of Independence, which was adopted by the Second Continental Congress on July 4, 1776. The term "United States" and its initialism "U.S.", used as nouns or as adjectives in English, are common short names for the country. The initialism "USA", a noun, is also common. "United States" and "U.S." are the established terms throughout the U.S. federal government, with prescribed rules. "The States" is an established colloquial shortening of the name, used particularly from abroad; "stateside" is the corresponding adjective or adverb. "America" is the feminine form of the first name of Americus Vesputius, the Latinized name of Italian explorer Amerigo Vespucci (1454–1512); it was first used as a place name by the German cartographers Martin Waldseemüller and Matthias Ringmann in 1507. Vespucci proposed that the West Indies discovered by Christopher Columbus in 1492 were part of a previously unknown landmass and not among the Indies at the eastern limit of Asia. In English, the term "America" (used without a qualifier) seldom refers to topics unrelated to the United States. "The Americas" is the general term to describe the totality of the continents of North and South America. 

History 

Indigenous peoples 

The first inhabitants of North America migrated from Siberia approximately 15,000 years ago, either across the Bering land bridge or along the now-submerged Ice Age coastline. The Clovis culture, which appeared around 11,000 BCE in North America, is believed to be the first widespread culture in the Americas. Over time, Indigenous North American cultures grew increasingly sophisticated, and some, such as the Mississippian culture, developed agriculture, architecture, and complex societies. In the post-archaic period, the Mississippian cultures were located in the midwestern, eastern, and southern regions, and the Algonquian in the Great Lakes region and along the Eastern Seaboard, while the Hohokam culture and Ancestral Puebloans inhabited the Southwest. Native population estimates of what is now the United States before the arrival of European colonizers range from around 500,000 to nearly 10 million. 

European exploration, colonization and conflict (1513–1765) 

Christopher Columbus began exploring the Caribbean for Spain in 1492, leading to Spanish-speaking settlements and missions from what are now Puerto Rico and Florida to New Mexico and California. The first Spanish colony in the present-day continental United States was Spanish Florida, chartered in 1513. After several settlements failed there due to starvation and disease, Spain's first permanent town, Saint Augustine, was founded in 1565. France established its own settlements in French Florida in 1562, but they were either abandoned (Charlesfort, 1578) or destroyed by Spanish raids (Fort Caroline, 1565). Permanent French settlements were founded much later along the Great Lakes (Fort Detroit, 1701), the Mississippi River (St. Louis, 1764) and especially the Gulf of Mexico (New Orleans, 1718). Early European colonies also included the thriving Dutch colony of New Nederland (settled 1626, present-day New York) and the small Swedish colony of New Sweden (settled 1638 in what became Delaware). British colonization of the East Coast began with the Virginia Colony (1607) and the Plymouth Colony (Massachusetts, 1620). The Mayflower Compact in Massachusetts and the Fundamental Orders of Connecticut established precedents for local representative self-governance and constitutionalism that would develop throughout the American colonies. While European settlers in what is now the United States experienced conflicts with Native Americans, they also engaged in trade, exchanging European tools for food and animal pelts. Relations ranged from close cooperation to warfare and massacres. The colonial authorities often pursued policies that forced Native Americans to adopt European lifestyles, including conversion to Christianity. Along the eastern seaboard, settlers trafficked Africans through the Atlantic slave trade, largely to provide manual labor on plantations. The original Thirteen Colonies that would later found the United States were administered as possessions of the British Empire by Crown-appointed governors, though local governments held elections open to most white male property owners. The colonial population grew rapidly from Maine to Georgia, eclipsing Native American populations; by the 1770s, the natural increase of the population was such that only a small minority of Americans had been born overseas. The colonies' distance from Britain facilitated the entrenchment of self-governance, and the First Great Awakening, a series of Christian revivals, fueled colonial interest in guaranteed religious liberty. 

American Revolution and the early republic (1765–1800) 

Following its victory in the French and Indian War, Britain began to assert greater control over local affairs in the Thirteen Colonies, resulting in growing political resistance. One of the primary grievances of the colonists was the denial of their rights as Englishmen, particularly the right to representation in the British government that taxed them. To demonstrate their dissatisfaction and resolve, the First Continental Congress met in 1774 and passed the Continental Association, a colonial boycott of British goods enforced by local "committees of safety" that proved effective. The British attempt to then disarm the colonists resulted in the 1775 Battles of Lexington and Concord, igniting the American Revolutionary War. At the Second Continental Congress, the colonies appointed George Washington Commander-in-Chief of the Continental Army, and created a committee that named Thomas Jefferson to draft the Declaration of Independence. Two days after the Second Continental Congress passed the Lee Resolution to create an independent, sovereign nation, the Declaration was adopted on July 4, 1776. The political values of the American Revolution evolved from an armed rebellion demanding reform within an empire to a revolution that created a new social and governing system founded on the defense of liberty and the protection of inalienable natural rights; equality under the law; sovereignty of the people; republicanism over monarchy, aristocracy, and other hereditary political power; civic virtue; and an intolerance of political corruption. The Founding Fathers of the United States, who included Washington, Jefferson, John Adams, Benjamin Franklin, Alexander Hamilton, John Jay, James Madison, Thomas Paine, and many others, were inspired by Classical, Renaissance, and Enlightenment philosophies and ideas. Though in practical effect since its drafting in 1777, the Articles of Confederation were ratified in 1781 and formally established a decentralized government that operated until 1789. After the British surrender at the siege of Yorktown in 1781, American sovereignty was internationally recognized by the Treaty of Paris (1783), through which the U.S. also gained territory stretching west to the Mississippi River, north to present-day Canada, and south to Spanish Florida. The Northwest Ordinance (1787) established the precedent by which the country's territory would expand with the admission of new states, rather than the expansion of existing states. The U.S. Constitution was drafted at the 1787 Constitutional Convention to overcome certain limitations of the Articles. It went into effect in 1789, creating a federal republic governed by three separate branches that together formed a system of checks and balances. George Washington was elected the country's first president under the Constitution, and the Bill of Rights (a series of ten amendments to the Constitution) was adopted in 1791 to allay skeptics' concerns about the power of the more centralized government. Washington's resignation as Commander-in-Chief after the Revolutionary War and his later refusal to run for a third term as the country's first president established a precedent for the supremacy of civil authority in the United States and the peaceful transfer of power. 

Westward expansion and Civil War (1800–1865) 

In the late 18th century, American settlers began to expand westward in larger numbers, many with a sense of manifest destiny. The Louisiana Purchase of 1803 from France nearly doubled the territory of the United States. Lingering issues with Britain remained, leading to the War of 1812, which was fought to a draw. Spain ceded Florida and its Gulf Coast territory in 1819. The Missouri Compromise of 1820, which admitted Missouri as a slave state and Maine as a free state, attempted to balance the desire of northern states to prevent the expansion of slavery into new territories with that of southern states to extend it there. Primarily, the compromise prohibited slavery in all other lands of the Louisiana Purchase north of the 36°30′ parallel. As Americans expanded further into territory inhabited by Native Americans, the federal government implemented policies of Indian removal or assimilation. The most significant such legislation was the Indian Removal Act of 1830, a key policy of President Andrew Jackson. It resulted in the Trail of Tears (1830–1850), in which an estimated 60,000 Native Americans living east of the Mississippi River were forcibly removed and displaced to lands far to the west, causing 13,200 to 16,700 deaths along the forced march. Settler expansion as well as this influx of Indigenous peoples from the East resulted in the American Indian Wars west of the Mississippi. During the colonial period, slavery became legal in all the Thirteen colonies, and by 1770 it provided the main labor force in the large-scale, agriculture-dependent economies of the Southern Colonies from Maryland to Georgia. The practice began to be significantly questioned during the American Revolution, and spurred by an active abolitionist movement that had reemerged in the 1830s, states in the North enacted laws to prohibit slavery within their boundaries. At the same time, support for slavery had strengthened in Southern states, with widespread use of inventions such as the cotton gin (1793) having made slavery immensely profitable for Southern elites. The United States annexed the Republic of Texas in 1845, and the 1846 Oregon Treaty led to U.S. control of the present-day American Northwest. Dispute with Mexico over Texas led to the Mexican–American War (1846–1848). After the victory of the U.S., Mexico recognized U.S. sovereignty over Texas, New Mexico, and California in the 1848 Mexican Cession; the cession's lands also included the future states of Nevada, Colorado and Utah. The California gold rush of 1848–1849 spurred a huge migration of white settlers to the Pacific coast, leading to even more confrontations with Native populations. One of the most violent, the California genocide of thousands of Native inhabitants, lasted into the mid-1870s. Additional western territories and states were created. 

Throughout the 1850s, the sectional conflict regarding slavery was further inflamed by national legislation in the U.S. Congress and decisions of the Supreme Court. In Congress, the Fugitive Slave Act of 1850 mandated the forcible return to their enslavers in the South of persons taking refuge in non-slave states, while the Kansas–Nebraska Act of 1854 effectively gutted the anti-slavery requirements of the Missouri Compromise. In its Dred Scott decision of 1857, the Supreme Court ruled against an enslaved person brought into non-slave territory, simultaneously declaring the entire Missouri Compromise to be unconstitutional. These and other events exacerbated tensions between North and South that would culminate in the American Civil War (1861–1865). Beginning with South Carolina, 11 slave-state governments voted to secede from the United States in 1861, joining to create the Confederate States of America. All other state governments remained loyal to the Union. War broke out in April 1861 after the Confederacy bombarded Fort Sumter. Following the Emancipation Proclamation on January 1, 1863, many freed slaves joined the Union army. The war began to turn in the Union's favor following the 1863 Siege of Vicksburg and Battle of Gettysburg, and the Confederates surrendered in 1865 after the Union's victory in the Battle of Appomattox Court House. 

Reconstruction, Gilded Age, and Progressive Era (1866–1917) 

Efforts toward reconstruction in the secessionist South had begun as early as 1862, but it was only after President Lincoln's assassination that the three Reconstruction Amendments to the Constitution were ratified to protect civil rights. The amendments codified nationally the abolition of slavery and involuntary servitude except as punishment for crimes, promised equal protection under the law for all persons, and prohibited discrimination on the basis of race or previous enslavement. As a result, African Americans took an active political role in ex-Confederate states in the decade following the Civil War. The former Confederate states were readmitted to the Union, beginning with Tennessee in 1866 and ending with Georgia in 1870. National infrastructure, including transcontinental telegraph and railroads, spurred growth in the American frontier. This was accelerated by the Homestead Acts, through which nearly 10 percent of the total land area of the United States was given away free to some 1.6 million homesteaders. From 1865 through 1917, an unprecedented stream of immigrants arrived in the United States, including 24.4 million from Europe. Most came through the Port of New York, as New York City and other major cities on the East Coast became home to large Jewish, Irish, and Italian populations. Many Northern Europeans as well as significant numbers of Germans and other Central Europeans moved to the Midwest. At the same time, about one million French Canadians migrated from Quebec to New England. During the Great Migration, millions of African Americans left the rural South for urban areas in the North. Alaska was purchased from Russia in 1867. The Compromise of 1877 is generally considered the end of the Reconstruction era, as it resolved the electoral crisis following the 1876 presidential election and led President Rutherford B. Hayes to reduce the role of federal troops in the South. Immediately, the Redeemers began evicting the Carpetbaggers and quickly regained local control of Southern politics in the name of white supremacy. African Americans endured a period of heightened, overt racism following Reconstruction, a time often considered the nadir of American race relations. A series of Supreme Court decisions, including Plessy v. Ferguson, emptied the Fourteenth and Fifteenth Amendments of their force, allowing Jim Crow laws in the South to remain unchecked, sundown towns in the Midwest, and segregation in communities across the country, which would be reinforced in part by the policy of redlining later adopted by the federal Home Owners' Loan Corporation. An explosion of technological advancement, accompanied by the exploitation of cheap immigrant labor, led to rapid economic expansion during the Gilded Age of the late 19th century. It continued into the early 20th, when the United States already outpaced the economies of Britain, France, and Germany combined. Tycoons led the nation's expansion in the railroad, petroleum, and steel industries, as the United States emerged as a pioneer of the automotive industry. This fostered the amassing of enormous economic and political power by a few prominent industrialists, largely through the formation of trusts and monopolies to prevent competition. These changes resulted in significant increases in economic inequality, slum conditions, and social unrest, creating a fertile environment for labor unions to flourish and, to a more limited extent, socialist movements. This period eventually ended with the advent of the Progressive Era, which was characterized by significant economic, legislative and social reforms. Pro-American elements in Hawaii overthrew the Hawaiian monarchy; the islands were annexed in 1898. That same year, Puerto Rico, the Philippines, and Guam were ceded to the U.S. by Spain after the latter's defeat in the Spanish–American War. (The Philippines was granted full independence from the U.S. on July 4, 1946, following World War II. Puerto Rico and Guam have remained U.S. territories.) American Samoa was acquired by the United States in 1900 after the Second Samoan Civil War. The U.S. Virgin Islands were purchased from Denmark in 1917 after Danish voters approved the sale in a 1916 referendum. 

World War I, Great Depression, and World War II (1917–1945) 

The United States entered World War I alongside the Allies in 1917 helping to turn the tide against the Central Powers. In 1920, a constitutional amendment granted nationwide women's suffrage. During the 1920s and 1930s, radio for mass communication and early television transformed communications nationwide. The Wall Street Crash of 1929 had triggered the Great Depression, to which President Franklin D. Roosevelt responded with the New Deal plan of "reform, recovery and relief", a series of unprecedented and sweeping recovery programs and employment relief projects combined with financial reforms and regulations. Initially neutral during World War II, the U.S. began supplying war materiel to the Allies of World War II in March 1941 and entered the war in December after Japan's attack on Pearl Harbor. Agreeing to a "Europe first" policy, the U.S. concentrated its wartime efforts on Japan's allies Italy and Germany until their final defeat in May 1945. The U.S. developed the first nuclear weapons and used them against the Japanese cities of Hiroshima and Nagasaki in August 1945, which historians consider central to the end of World War II in Asia. The United States was one of the "Four Policemen" who met to plan the post-war world, alongside the United Kingdom, the Soviet Union, and China. The U.S. emerged relatively unscathed from the war, with even greater economic power and international political influence. 

Cold War and social revolution (1945–1991) 

The end of World War II in 1945 left the U.S. and the Soviet Union as superpowers, each with its own political, military, and economic sphere of influence. Geopolitical tensions between the two superpowers soon led to the Cold War. The U.S. implemented a policy of containment intended to limit the Soviet Union's sphere of influence; engaged in regime change against governments perceived to be aligned with the Soviets; and prevailed in the Space Race, which culminated with the first crewed Moon landing in 1969. Domestically, the U.S. experienced economic growth, urbanization, and population growth following World War II. The civil rights movement emerged, with Martin Luther King Jr. becoming a prominent leader in the early 1960s. The Great Society plan of President Lyndon B. Johnson's administration resulted in groundbreaking and broad-reaching laws, policies and a constitutional amendment to counteract some of the worst effects of lingering institutional racism. The counterculture movement in the U.S. brought significant social changes, including the liberalization of attitudes toward recreational drug use and sexuality. It also encouraged open defiance of the military draft (leading to the end of conscription in 1973) and wide opposition to U.S. intervention in Vietnam, with the U.S. totally withdrawing in 1975. A societal shift in the roles of women was significantly responsible for the large increase in female paid labor participation starting in the 1970s, and by 1985 the majority of American women aged 16 and older were employed. The Fall of Communism and the dissolution of the Soviet Union from 1989 to 1991 marked the end of the Cold War and left the United States as the world's sole superpower. This cemented the United States' global influence, reinforcing the concept of the "American Century" as the U.S. dominated international political, cultural, economic, and military affairs. 

Contemporary (1991–present) 

The 1990s saw the longest recorded economic expansion in American history, a dramatic decline in U.S. crime rates, and advances in technology. Throughout this decade, technological innovations such as the World Wide Web, the evolution of the Pentium microprocessor in accordance with Moore's law, rechargeable lithium-ion batteries, the first gene therapy trial, and cloning either emerged in the U.S. or were improved upon there. In the 1990s, the Human Genome Project was launched, while Nasdaq became the first stock market in the United States to trade online. In the Gulf War of 1991, an American-led international coalition of states expelled an Iraqi invasion force that had occupied neighboring Kuwait. The September 11 attacks on the United States in 2001 by the pan-Islamist militant organization al-Qaeda led to the U.S. launching the war on terror and military interventions in Afghanistan and Iraq. The U.S. housing bubble culminated in 2007 with the Great Recession, the largest economic contraction since the Great Depression. Beginning in the 2010s, and accelerating in the 2020s, the United States has experienced increased political polarization and significant democratic backsliding. The country's polarization was violently reflected in the January 2021 Capitol attack, when a mob of insurrectionists supporting President Donald Trump entered the U.S. Capitol and sought to prevent the peaceful transfer of power in an attempted self-coup d'état. It was the first presidential refusal of the peaceful transfer of power. Scholars have broadly classified the United States as transitioning from a liberal democratic polity to a hybrid regime since 2025. 

Geography 

The United States is the world's third-largest country by total area behind Russia and Canada. The 48 contiguous states and the District of Columbia have a combined area of 3,119,885 square miles (8,080,470 km2). In 2021, the United States had 8% of the Earth's permanent meadows and pastures and 10% of its cropland. Starting in the east, the coastal plain of the Atlantic seaboard gives way to inland forests and rolling hills in the Piedmont plateau region. The Appalachian Mountains and the Adirondack Massif separate the East Coast from the Great Lakes and the grasslands of the Midwest. The Mississippi River System, the world's fourth-longest river system, runs predominantly north–south through the center of the country. The flat and fertile prairie of the Great Plains stretches to the west, interrupted by a highland region in the southeast. 

The Rocky Mountains, west of the Great Plains, extend north to south across the country, peaking at over 14,000 feet (4,300 m) in Colorado. The supervolcano underlying Yellowstone National Park in the Rocky Mountains, the Yellowstone Caldera, is the continent's largest volcanic feature. Farther west are the rocky Great Basin and the Chihuahuan, Sonoran, and Mojave deserts. In the northwest corner of Arizona, carved by the Colorado River, is the Grand Canyon, a steep-sided canyon and popular tourist destination known for its overwhelming visual size and intricate, colorful landscape. The Cascade and Sierra Nevada mountain ranges run close to the Pacific coast. The lowest and highest points in the contiguous United States are in the State of California, about 84 miles (135 km) apart. At an elevation of 20,310 feet (6,190.5 m), Alaska's Denali (also called Mount McKinley) is the highest peak in the country and on the continent. Active volcanoes in the U.S. are common throughout Alaska's Alexander and Aleutian Islands. Located entirely outside North America, the archipelago of Hawaii consists of volcanic islands, physiographically and ethnologically part of the Polynesian subregion of Oceania. In addition to its total land area, the United States has one of the world's largest marine exclusive economic zones spanning approximately 4.5 million square miles (11.7 million km2) of ocean. 

Climate 

With its large size and geographic variety, the United States includes most climate types. East of the 100th meridian, the climate ranges from humid continental in the north to humid subtropical in the south. The western Great Plains are semi-arid. Many mountainous areas of the American West have an alpine climate. The climate is arid in the Southwest, Mediterranean in coastal California, and oceanic in coastal Oregon, Washington, and southern Alaska. Most of Alaska is subarctic or polar. Hawaii, the southern tip of Florida and U.S. territories in the Caribbean and Pacific are tropical. The United States receives more high-impact extreme weather incidents than any other country. States bordering the Gulf of Mexico are prone to hurricanes, and most of the world's tornadoes occur in the country, mainly in Tornado Alley. Due to climate change, extreme weather has become more frequent in the U.S. in the 21st century, with three times the number of reported heat waves compared to the 1960s. Since the 1990s, droughts in the American Southwest have become more persistent and more severe. The regions considered as the most attractive to the population are the most vulnerable. 

Biodiversity and conservation 

The U.S. is one of 17 megadiverse countries containing large numbers of endemic species: about 17,000 species of vascular plants occur in the contiguous United States and Alaska, and over 1,800 species of flowering plants are found in Hawaii, few of which occur on the mainland. The United States is home to 428 mammal species, 784 birds, 311 reptiles, 295 amphibians, and around 91,000 insect species. There are 63 national parks, and hundreds of other federally managed monuments, forests, and wilderness areas, administered by the National Park Service and other agencies. About 28% of the country's land is publicly owned and federally managed, primarily in the Western States. Most of this land is protected, though some is leased for commercial use, and less than one percent is used for military purposes. Environmental issues in the United States include debates on non-renewable resources and nuclear energy, air and water pollution, biodiversity, logging and deforestation, and climate change. The U.S. Environmental Protection Agency (EPA) is the federal agency charged with addressing most environmental-related issues. The idea of wilderness has shaped the management of public lands since 1964, with the Wilderness Act. The Endangered Species Act of 1973 provides a way to protect threatened and endangered species and their habitats. The United States Fish and Wildlife Service implements and enforces the Act. In 2024, the U.S. ranked 35th among 180 countries in the Environmental Performance Index. 

Government and politics 

The United States is a federal republic consisting of 50 states and a federal capital district, Washington, D.C. The U.S. asserts sovereignty over five unincorporated territories and several uninhabited island possessions. It is the world's oldest surviving federation, and its presidential system of federal government has been adopted, in whole or in part, by many newly independent states worldwide following their decolonization. The Constitution of the United States serves as the country's supreme legal document. The United States was the most prominent liberal democracy for much of the 20th and early 21st centuries, but has undergone significant democratic backsliding and a shift toward a hybrid regime—a political system combining autocratic and democratic features. Gerrymandering, the manipulation of districts for political advantage, is widespread and has been driven by 2010s and 2020s Supreme Court rulings. There is an ongoing debate among political scientists on whether the country is more appropriately classified as an electoral autocracy or illiberal democracy, with few still considering it to meet the criteria of a robust liberal democracy. 

Federal government 

Composed of three branches, all headquartered in Washington, D.C., the federal government is the national government of the United States. The U.S. Constitution establishes a separation of powers intended to provide a system of checks and balances to prevent any of the three branches from becoming supreme. The three-branch system is known as the presidential system, in contrast to the parliamentary system where the executive is part of the legislative body. Many countries around the world adopted this aspect of the 1789 Constitution of the United States, especially in the postcolonial Americas. 

Legislature The U.S. Congress is a bicameral legislature made up of the Senate and the House of Representatives. The Senate has 100 members—two residents from each state and elected by that state's voters for a six-year term. The House of Representatives has 435 members, elected for a two-year term by the constituency of the congressional district where they reside. A state's legislature decides the district boundaries, which are contiguous within the state. Every U.S. congressional district is of equivalent population and sends one representative to Congress. Election years for senators are staggered so that only one-third of them will be up for election every two years. U.S. representatives are all up for election at the same time every two years. The U.S. Congress makes federal law, declares war, approves treaties, has the power of the purse, and has the power of impeachment. One of its foremost non-legislative functions is the power to investigate and oversee the executive branch. Congressional oversight is usually delegated to committees and is facilitated by Congress' power to issue subpoenas. Much of the work of Congress is performed by a collection of committees, each appointed for a specific purpose or function. Committee membership is by tradition and statute bipartisan, but all committees are chaired by a member of the majority party, who sets the committee agenda. 

Executive 

The U.S. president is the head of state, commander-in-chief of the military, and chief executive of the federal government. The president appoints the members of the Cabinet, subject to Senate approval, and names other officials who administer and enforce federal law and policy through their respective agencies. The president has the ability to veto legislative bills from the U.S. Congress before they become law. However, presidential vetoes can be overridden by a two-thirds supermajority vote in both chambers of Congress. The president also has clemency power for federal crimes and can issue pardons. Finally, the president has the authority to issue expansive "executive orders" in a number of policy areas, subject to judicial review. Candidates for president typically campaign with a vice-presidential running mate, and both candidates are typically elected together, or defeated together, in a presidential election. Unlike other votes in American politics, this is technically an indirect election in which the winner will be determined by the U.S. Electoral College. There, votes are officially cast by individual electors selected by their state legislature. In practice, however, each of the 50 states chooses a group of presidential electors who are required by state law to confirm the winner of their state's popular vote. Each state is allocated two electors plus one additional elector for every congressional district in the state, which in effect combines to equal the number of elected officials that state sends to Congress. The District of Columbia, with no representatives or senators, is allocated three electoral votes. Both the president and the vice president serve a four-year term, and the president may be reelected to the office only once, for one additional four-year term. 

Judiciary The U.S. federal judiciary, whose judges are all appointed for life by the president with Senate approval, consists primarily of the U.S. Supreme Court, the U.S. courts of appeals, and the U.S. district courts. The lowest level in the federal judiciary is the federal district court, which decides all cases considered to be under "original jurisdiction", such as federal statutes, constitutional law, or international treaties. After a federal district court has decided a case, its ruling may be contested and sent to a higher court, a federal court of appeals. The U.S. judicial system's 12 federal circuits divide the country into 12 separate geographic administrative regions for appeals decisions. The next and highest court in the system is the Supreme Court of the United States. The U.S. Supreme Court interprets laws and overturns those it finds unconstitutional. On average, the Supreme Court receives about 7,000 appeals petitions for writs of certiorari each year, but only grants about 80. Consisting of nine members led by the Chief Justice of the United States, the court usually judges each case before it by majority decision. In the case of an impeachment trial in the Senate of a sitting president, the chief justice presides. As with all other federal judges, the members are appointed for life by the sitting president with Senate approval when a vacancy becomes available. 

Subdivisions 

In the U.S. federal system, sovereign powers are shared between three levels of government specified in the Constitution: the federal government, the states, and Indian tribes. The U.S. also asserts sovereignty over five permanently inhabited territories: American Samoa, Guam, the Northern Mariana Islands, Puerto Rico, and the U.S. Virgin Islands. Residents of the 50 states are governed by their elected state government, under state constitutions compatible with the national constitution, and by elected local governments that are administrative divisions of a state. States are subdivided into counties or county equivalents, which (except in Hawaii) can permit the formation of independent municipalities administered by their own elected representatives. The District of Columbia is a federal district containing the U.S. capital, Washington, D.C. The federal district is an administrative division of the federal government. 

Indian country is made up of 574 federally recognized tribes and 326 Indian reservations. They hold a government-to-government relationship with the U.S. federal government in Washington and are legally defined as domestic dependent nations with inherent tribal sovereignty rights. In addition to the five major territories, the U.S. also asserts sovereignty over the United States Minor Outlying Islands in the Pacific Ocean and the Caribbean. The seven undisputed islands without permanent populations are Baker Island, Howland Island, Jarvis Island, Johnston Atoll, Kingman Reef, Midway Atoll, and Palmyra Atoll. U.S. sovereignty over the unpopulated Bajo Nuevo Bank, Navassa Island, Serranilla Bank, and Wake Island is disputed. 

Political parties 

The Constitution is silent on political parties. However, they developed independently in the 18th century with the Federalist and Anti-Federalist parties. Since then, the United States has operated as a de facto two-party system, though the parties have changed over time. Since the mid-19th century, the two main national parties have been the Democratic Party and the Republican Party. Political scientists and comparative political studies classify the modern Democratic Party as a liberal party whose political platform stands near the ideological center of the left–right political spectrum, and the modern Republican Party as a right-wing populist and nationalist party whose political platform is right-wing to far-right. 

Foreign relations 

The United States has an established structure of foreign relations, with the world's second-largest diplomatic corps as of 2024. It is a permanent member of the United Nations Security Council and home to the United Nations headquarters. The United States is a member of the G7, G20, and OECD intergovernmental organizations. Almost all countries have embassies and many have consulates (official representatives) in the country. Likewise, nearly all countries host formal diplomatic missions with the United States, except Iran, North Korea, and Bhutan. Though Taiwan does not have formal diplomatic relations with the U.S., it maintains close unofficial relations. The United States regularly supplies Taiwan with military equipment to deter potential Chinese aggression. The country's geopolitical attention has increasingly turned to the Indo-Pacific, where the U.S. joined the Quadrilateral Security Dialogue and AUKUS. The United States has a "Special Relationship" with the United Kingdom and strong ties with Canada, Australia, New Zealand, the Philippines, Japan, South Korea, Israel, and several European Union countries such as France, Italy, Germany, Spain, and Poland. The U.S. works closely with its NATO allies on military and national security issues, and with countries in the Americas through the Organization of American States and the United States–Mexico–Canada Free Trade Agreement. The U.S. exercises full international defense authority and responsibility for Micronesia, the Marshall Islands, and Palau through the Compact of Free Association. It has increasingly conducted strategic cooperation with India, while its ties with China have steadily deteriorated. Beginning in 2014, the U.S. had become a key ally of Ukraine. 

Military 

The president is the commander-in-chief of the United States Armed Forces and appoints its leaders, the secretary of defense and the Joint Chiefs of Staff. The Department of Defense, headquartered at the Pentagon near Washington, D.C., administers five of the six service branches, which are made up of the U.S. Army, Marine Corps, Navy, Air Force, and Space Force. The Coast Guard is administered by the Department of Homeland Security in peacetime and can be transferred to the Department of the Navy in wartime. Total strength of the entire military is about 1.3 million active duty with an additional 400,000 in reserve. The United States military is widely regarded as the most powerful and advanced in the world. The U.S. spent $954 billion on its military in 2025, which is by far the largest amount of any country, making up 33% of global military spending and accounting for 3.1% of the country's GDP. The U.S. possesses 42% of the world's nuclear weapons—the second-largest stockpile after that of Russia. The United States has the third-largest combined armed forces in the world, behind the Chinese People's Liberation Army and Indian Armed Forces. In addition to the vast network of military bases on its soil, the U.S. maintains approximately 800 other bases and installations around the world, and it deploys greater than 100 active-duty personnel in each of 25 foreign countries. The United States has engaged in over 400 military interventions since its founding in 1776, with over half of these occurring between 1950 and 2019 and 25% occurring in the post–Cold War era. State defense forces (SDFs) are military units that operate under the sole authority of a state government. SDFs are authorized by state and federal law but are under the command of the state's governor. By contrast, the 54 U.S. National Guard organizations fall under the dual control of state or territorial governments and the federal government; their units can also become federalized entities, but SDFs cannot be federalized. The National Guard personnel of a state or territory can be federalized by the president under the National Defense Act Amendments of 1933; this legislation created the Guard and provides for the integration of Army National Guard and Air National Guard units and personnel into the U.S. Army and (since 1947) the U.S. Air Force. The total number of National Guard members is about 430,000, while the estimated combined strength of SDFs is less than 10,000. 

Law enforcement and criminal justice 

There are about 18,000 U.S. police agencies from local to national level in the United States. Law in the United States is mainly enforced by local police departments and sheriff departments in their municipal or county jurisdictions. The state police departments have authority in their respective state, and federal agencies such as the Federal Bureau of Investigation (FBI) and the U.S. Marshals Service have national jurisdiction and specialized duties, such as protecting civil rights, national security, enforcing U.S. federal courts' rulings and federal laws, and interstate criminal activity. State courts conduct almost all civil and criminal trials, while federal courts adjudicate the much smaller number of civil and criminal cases that relate to federal law. There is no unified "criminal justice system" in the United States. The American prison system is largely heterogenous, with thousands of relatively independent systems operating across federal, state, local, and tribal levels. In 2026, "these systems hold nearly 2 million people in 1,566 state prisons, 98 federal prisons, 3,116 local jails, 1,277 juvenile correctional facilities, 220 immigration detention facilities, and 77 Indian country jails, as well as in military prisons, civil commitment centers, state psychiatric hospitals, and prisons in the U.S. territories—at a system-wide cost of at least $445 billion each year." Despite disparate systems of confinement, four main institutions dominate: federal prisons, state prisons, local jails, and juvenile correctional facilities. Federal prisons are run by the Federal Bureau of Prisons and hold pretrial detainees as well as people who have been convicted of federal crimes. State prisons, run by the department of corrections of each state, hold people sentenced and serving prison time (usually longer than one year) for felony offenses. Local jails are county or municipal facilities that incarcerate defendants prior to trial; they also hold those serving short sentences (typically under a year). Juvenile correctional facilities are operated by local or state governments and serve as longer-term placements for any minor adjudicated as delinquent and ordered by a judge to be confined. In January 2023, the United States had the sixth-highest per capita incarceration rate in the world—531 people per 100,000 inhabitants—and the largest prison and jail population in the world, with more than 1.9 million people incarcerated. An analysis of the World Health Organization Mortality Database from 2010 showed U.S. homicide rates "were 7 times higher than in other high-income countries, driven by a gun homicide rate that was 25 times higher". The country's legal system is classified by some political scientists and legal scholars as operating under a "partial rule of law" system, according to two 2026 surveys conducted by Bright Line Watch and the London School of Economics. 

Economy 

The U.S. has a highly developed mixed economy that has been the world's largest nominally since about 1890. Its 2024 gross domestic product (GDP) of more than $29 trillion constituted over 25% of nominal global economic output, or 15% at purchasing power parity (PPP). From 1983 to 2008, U.S. real compounded annual GDP growth was 3.3%, compared to a 2.3% weighted average for the rest of the G7. The country ranks first in the world by nominal GDP, second when adjusted for purchasing power parities (PPP), and ninth by PPP-adjusted GDP per capita. In August 2026, the total U.S. federal government debt passed $40 trillion, having more than doubled in the previous decade. 

Of the world's 500 largest companies by revenue, 138 were headquartered in the U.S. in 2025, the highest number of any country. The U.S. dollar is the currency most used in international transactions and the world's foremost reserve currency, backed by the country's dominant economy, its military, the petrodollar system, its large U.S. treasuries market, and its linked eurodollar. Several countries use it as their official currency, and in others it is the de facto currency. The U.S. has free trade agreements with several countries, including the USMCA. Although the United States has reached a post-industrial level of economic development and is often described as having a service economy, it remains a major industrial power; in 2024, the U.S. manufacturing sector was the world's second-largest by value output after China's. 

New York City is the world's principal financial center, and its metropolitan area is the world's largest metropolitan economy. The New York Stock Exchange and Nasdaq, both located in New York City, are the world's two largest stock exchanges by market capitalization and trade volume. The United States is at the forefront of technological advancement and innovation in many economic fields, especially in artificial intelligence; electronics and computers; pharmaceuticals; and medical, aerospace and military equipment. The country's economy is fueled by abundant natural resources, a well-developed infrastructure, and high productivity. The largest trading partners of the United States are the European Union, Mexico, Canada, China, Japan, South Korea, the United Kingdom, Vietnam, India, and Taiwan. The United States is the world's largest importer and second-largest exporter. It is by far the world's largest exporter of services. Americans have the highest average household and employee income among OECD member states, and the fourth-highest median household income in 2023, up from sixth-highest in 2013. With personal consumption expenditures of over $18.5 trillion in 2023, the U.S. has a heavily consumer-driven economy and is the world's largest consumer market. The U.S. ranked first in the number of dollar billionaires and millionaires in 2023, with 735 billionaires and nearly 22 million millionaires. Wealth in the United States is highly concentrated; in 2011, the richest 10% of the adult population owned 72% of the country's household wealth, while the bottom 50% owned just 2%. U.S. wealth inequality increased substantially since the late 1980s, and income inequality in the U.S. reached a record high in 2019. In 2024, the country had some of the highest wealth and income inequality levels among OECD countries. Since the 1970s, there has been a decoupling of U.S. wage gains from worker productivity, while the economy has become more dominated by financial services and stock trading. In 2016, the top fifth of earners took home more than half of all income, giving the U.S. one of the widest income distributions among OECD countries. There were about 771,480 homeless persons in the U.S. in 2024. In 2022, 6.4 million children experienced food insecurity. Feeding America estimates that around one in five, or approximately 13 million, children experience hunger in the U.S. and do not know where or when they will get their next meal. Also in 2022, about 37.9 million people, or 11.5% of the U.S. population, were living in poverty. The United States has a smaller welfare state and redistributes less income through government action than most other high-income countries. It is the only advanced economy that does not guarantee its workers paid vacation nationally and one of a few countries in the world without federal paid family leave as a legal right. The United States has a higher percentage of low-income workers than almost any other developed country, largely because of a weak collective bargaining system and lack of government support for at-risk workers. 

Science and technology 

The United States has been a leader in technological innovation since the late 19th century and scientific research since the mid-20th century. Methods for producing interchangeable parts and the establishment of a machine tool industry enabled the large-scale manufacturing of U.S. consumer products in the late 19th century. By the early 20th century, factory electrification, the introduction of the assembly line, and other labor-saving techniques created the system of mass production. In the 21st century, the United States continues to be one of the world's foremost scientific powers, though China has emerged as a major competitor in many fields. The U.S. has the highest research and development expenditures of any country and ranks ninth as a percentage of GDP. In 2022, the United States was (after China) the country with the second-highest number of published scientific papers. In 2021, the U.S. ranked second (also after China) by the number of patent applications, and third by trademark and industrial design applications (after China and Germany), according to World Intellectual Property Indicators. In 2025 the United States ranked third (after Switzerland and Sweden) in the Global Innovation Index. The United States is considered to be a world leader in the development of artificial intelligence technology. In 2023, the United States was ranked the second most technologically advanced country in the world (after South Korea) by Global Finance magazine. 

Spaceflight 

The United States has maintained a space program since the late 1950s, beginning with the establishment of the National Aeronautics and Space Administration (NASA) in 1958. In 1961, the United States became the second country (after the Soviet Union) to successfully launch a human into space. NASA's Apollo program (1961–1972) achieved the first crewed Moon landing with the 1969 Apollo 11 mission; it remains one of the agency's most significant milestones. Other major endeavors by NASA include the Space Shuttle program (1981–2011), the Voyager program (1972–present), the Hubble and James Webb space telescopes (launched in 1990 and 2021, respectively), and the multi-mission Mars Exploration Program (Spirit, Opportunity, Curiosity, and Perseverance). NASA is one of five agencies collaborating on the International Space Station (ISS); U.S. contributions to the ISS include several modules, including Destiny (2001), Harmony (2007), and Tranquility (2010), as well as ongoing logistical and operational support. The United States private sector dominates the global commercial spaceflight industry. Prominent American spaceflight contractors include Blue Origin, Boeing, Lockheed Martin, Northrop Grumman, and SpaceX. NASA programs such as the Commercial Crew Program, Commercial Resupply Services, Commercial Lunar Payload Services, and NextSTEP have facilitated growing private-sector involvement in American spaceflight. 

Energy 

In 2023, the United States received approximately 84% of its energy from fossil fuel, and its largest source of energy was petroleum (38%), followed by natural gas (36%), renewable sources (9%), coal (9%), and nuclear power (9%). In 2022, the United States constituted about 4% of the world's population, but consumed around 16% of the world's energy. The U.S. ranks as the second-highest emitter of greenhouse gases behind China. The U.S. is the world's largest producer of nuclear power, generating around 30% of the world's nuclear electricity. It also has the highest number of nuclear power reactors of any country. From 2024, the U.S. plans to triple its nuclear power capacity by 2050. 

Transportation 

The United States' 4 million miles (6.4 million kilometers) of road network, owned almost entirely by state and local governments, is the longest in the world. The extensive Interstate Highway System that connects all major U.S. cities is funded mostly by the federal government but maintained by state departments of transportation. The system is further extended by state highways and some private toll roads. The U.S. is among the ten countries with the highest vehicle ownership per capita (850 vehicles per 1,000 people) in 2022. A 2022 study found that 76% of U.S. commuters drive alone and 14% ride a bicycle, including bike owners and users of bike-sharing networks. About 11% use some form of public transportation. Public transportation in the United States is well developed in the largest urban areas, notably New York City, Washington, D.C., Boston, Philadelphia, Chicago, and the San Francisco Bay Area; otherwise, coverage is generally less extensive than in most other developed countries. The U.S. also has many relatively car-dependent localities. Long-distance intercity travel is provided primarily by airlines, but travel by rail is more common along the Northeast Corridor, where the only high-speed rail in the U.S. that meets international standards operates. Amtrak, the country's government-sponsored national passenger rail company, has a relatively sparse network compared to that of Western European countries. Service is concentrated in the Northeast, California, the Midwest, and the Pacific Northwest. The country's rail transport network, the longest in the world at 182,412.3 mi (293,564.2 km), handles mostly freight (in contrast to more passenger-centered rail in Europe). Because they are often privately owned, U.S. railroads lag behind much of the rest of the world in terms of electrification. 

The United States has an extensive air transportation network. U.S. civilian airlines are all privately owned. The three largest airlines in the world, by total number of passengers carried, are U.S.-based; American Airlines became the global leader after its 2013 merger with US Airways. Of the 50 busiest airports in the world, 16 are in the United States, as well as five of the top 10. The world's busiest airport by passenger volume is Hartsfield–Jackson Atlanta International in Atlanta, Georgia. In 2022, most of the 19,969 U.S. airports were owned and operated by local government authorities, and there are also some private airports. Some 5,193 are designated as "public use", including for general aviation. The Transportation Security Administration (TSA) has provided security at most major airports since 2001. The country's inland waterways are the world's fifth-longest, totaling 25,482 mi (41,009 km). They are used extensively for freight, recreation, and a small amount of passenger traffic. Of the world's 50 busiest container ports, four are located in the United States, with the busiest in the country being the Port of Los Angeles. 

Demographics 

Population 

The U.S. Census Bureau reported 331,449,281 residents on April 1, 2020, making the United States the third-most-populous country in the world, after India and China. The Census Bureau's official 2025 population estimate was 341,784,857, an increase of 3.1% since the 2020 census. According to the Bureau's U.S. Population Clock, on July 1, 2024, the U.S. population had a net gain of one person every 16 seconds, or about 5400 people per day. In 2023, 51% of Americans age 15 and over were married, 6% were widowed, 10% were divorced, and 34% had never been married. In 2023, the total fertility rate for the U.S. stood at 1.6 children per woman, and, at 23%, it had the world's highest rate of children living in single-parent households in 2019. Most Americans live in the suburbs of major metropolitan areas. The United States has a diverse population; 37 ancestry groups have more than one million members. White Americans with ancestry from Europe, the Middle East, or North Africa form the largest racial and ethnic group at 57.8% of the United States population. Hispanic and Latino Americans form the second-largest group and are 18.7% of the United States population. African Americans constitute the country's third-largest ancestry group and are 12.1% of the total U.S. population. Asian Americans are the country's fourth-largest group, composing 5.9% of the United States population. The country's 3.7 million Native Americans account for about 1%, and some 574 native tribes are recognized by the federal government. In 2024, the median age of the United States population was 39.1 years. 

Urbanization 

About 82% of Americans live in metropolitan areas, particularly in suburbs and outer-ring exurbs; about half of those reside in cities with populations over 50,000. In 2024, 346 incorporated U.S. municipalities had populations over 100,000, 11 cities had more than one million residents, and four cities—New York City, Los Angeles, Chicago, and Houston—had populations exceeding two million. Some 56 U.S. metropolitan areas have one million or more residents. More recently, the fastest-growing metropolitan areas were in the South, while southern metros along the Mexican border and Gulf Coast metros susceptible to hurricanes declined the most in 2025. The New York metro area, which gained the most new residents in 2024, fell to 13th in 2025, due to a fall in immigrants. The top metro areas with rising populations in 2025 were Houston and Dallas–Fort Worth, followed by Atlanta, Phoenix and Charlotte. 

Language 

While many languages and dialects are spoken in the United States, English is by far the most commonly spoken and written. De facto, English is the official language of the United States, and in 2025, Executive Order 14224 declared English official. However, the U.S. has never had a statutory official language, as Congress has never passed a law to designate English as official for all three federal branches. Some laws, such as U.S. naturalization requirements, nonetheless standardize English. Twenty-eight states and the United States Virgin Islands have laws that designate English as the sole official language; 19 states and the District of Columbia have no official language. Three states and four U.S. territories have recognized local or indigenous languages in addition to English: Hawaii (Hawaiian), Alaska (twenty Native languages), South Dakota (Sioux), American Samoa (Samoan), Puerto Rico (Spanish), Guam (Chamorro), and the Northern Mariana Islands (Carolinian and Chamorro). In total, 169 Native American languages are spoken in the United States. In Puerto Rico, Spanish is more widely spoken than English. According to the American Community Survey (2020), some 245.4 million people in the U.S. age five and older spoke only English at home. About 41.2 million spoke Spanish at home, making it the second most commonly used language. Other languages spoken at home by one million people or more include Chinese (3.40 million), Tagalog (1.71 million), Vietnamese (1.52 million), Arabic (1.39 million), French (1.18 million), Korean (1.07 million), and Russian (1.04 million). German, spoken by 1 million people at home in 2010, fell to 881,000 estimated total speakers in 2020. 

Immigration 

America's immigrant population is by far the world's largest in absolute terms. In 2022, there were 87.7 million immigrants and U.S.-born children of immigrants in the United States, accounting for nearly 27% of the overall U.S. population. In 2017, out of the U.S. foreign-born population, some 45% (20.7 million) were naturalized citizens, 27% (12.3 million) were lawful permanent residents, 6% (2.2 million) were temporary lawful residents, and 23% (10.5 million) were unauthorized immigrants. In 2019, the top countries of origin for immigrants were Mexico (24% of immigrants), India (6%), China (5%), the Philippines (4.5%), and El Salvador (3%). In fiscal year 2022, over one million immigrants (most of whom entered through family reunification) were granted legal residence. The undocumented immigrant population in the U.S. reached a record high of 14 million in 2023. 

Religion 

The First Amendment guarantees the free exercise of religion in the country and forbids Congress from passing laws respecting its establishment. Religious practice is widespread, among the most diverse in the world, and profoundly vibrant. The country has the world's largest Christian population, which includes the fourth-largest population of Catholics. Other notable faiths include Judaism, Buddhism, Hinduism, Islam, New Age, and Native American religions. Religious practice varies significantly by region. "Ceremonial deism" is common in American culture. The overwhelming majority of Americans believe in a higher power or spiritual force, engage in spiritual practices such as prayer, and consider themselves religious or spiritual. In the Southern United States' "Bible Belt", evangelical Protestantism plays a significant role culturally; New England and the Western United States tend to be more secular. Mormonism, a Restorationist movement founded in the U.S. in 1847, is the predominant religion in Utah and a major religion in Idaho. 

Health 

According to the Centers for Disease Control and Prevention (CDC), average U.S. life expectancy at birth reached 79.0 years in 2024, its highest recorded level and an increase of 0.6 years over 2023. The CDC attributed the improvement to a significant fall in the number of fatal drug overdoses in the country, noting that "heart disease continues to be the leading cause of death in the United States, followed by cancer and unintentional injuries." In 2024, life expectancy at birth for American men rose to 76.5 years (+0.7 years compared to 2023), while life expectancy for women was 81.4 years (+0.3 years). Starting in 1998, life expectancy in the U.S. fell behind that of other wealthy industrialized countries, and Americans' "health disadvantage" gap has been increasing ever since. The Commonwealth Fund reported in 2020 that the U.S. had the highest suicide rate among high-income countries. Approximately one-third of the U.S. adult population is obese, and another third is overweight. The U.S. healthcare system far outspends that of any other country, measured both in per capita spending and as a percentage of GDP, but attains worse healthcare outcomes when compared to peer countries for reasons that are debated. The United States is the only developed country without a system of universal healthcare, and a significant proportion of the population that does not carry health insurance. Government-funded healthcare coverage for the poor (Medicaid) and for those age 65 and older (Medicare) is available to Americans who meet the programs' income or age qualifications. In 2010, President Barack Obama passed the Patient Protection and Affordable Care Act. Since the 2022 U.S. Supreme Court decision Dobbs v. Jackson Women's Health Organization, which effectively overruled Roe v. Wade (1973), abortion in the United States is no longer federally protected but is subject to the laws of each state or territory. 

Education 

American primary and secondary education, known in the U.S. as K–12 ("kindergarten through 12th grade"), is decentralized. School systems are operated by state, territorial, and sometimes municipal governments and regulated by the U.S. Department of Education. In general, children are required to attend school or an approved homeschool from the age of five or six (kindergarten or first grade) until they are 18 years old. This often brings students through the 12th grade, the final year of a U.S. high school, but some states and territories allow them to leave school earlier, at age 16 or 17. The U.S. spends more on education per student than any other country, an average of $18,614 per year per public elementary and secondary school student in 2020–2021. Among Americans age 25 and older, 92.2% graduated from high school, 62.7% attended some college, 37.7% earned a bachelor's degree, and 14.2% earned a graduate degree. The U.S. literacy rate is near-universal. The U.S. has produced the most Nobel Prize winners of any country, with 411 (having won 413 awards). U.S. tertiary or higher education has earned a global reputation. Many of the world's top universities, as listed by various ranking organizations, are in the United States, including 19 of the top 25. American higher education is dominated by state university systems, although the country's many private universities and colleges enroll about 20% of all American students. Local community colleges generally offer open admissions, lower tuition, and coursework leading to a two-year associate degree or a non-degree certificate. As for public expenditures on higher education, the U.S. spends more per student than the OECD average, and Americans spend more than all nations in combined public and private spending. Colleges and universities directly funded by the federal government do not charge tuition and are limited to military personnel and government employees, including: the U.S. service academies, the Naval Postgraduate School, and military staff colleges. Despite some student loan forgiveness programs in place, student loan debt increased by 102% between 2010 and 2020, and exceeded $1.7 trillion in 2022. 

Culture and society 

The United States is home to a wide variety of ethnic groups, traditions, and customs. The country has been described as having the values of individualism and personal autonomy, as well as a strong work ethic and competitiveness. Voluntary altruism toward others also plays a major role; according to a 2016 study by the Charities Aid Foundation, Americans donated 1.44% of total GDP to charity—the highest rate in the world by a large margin. Americans have traditionally been characterized by a unifying political belief in an "American Creed" that emphasizes consent of the governed, liberty, equality under the law, democracy, social equality, property rights, and a preference for limited government. The U.S. has acquired significant hard and soft power through its diplomatic influence, economic power, military alliances, and cultural exports such as American movies, music, video games, sports, and food. The influence that the United States exerts on other countries through soft power is referred to as Americanization. Nearly all present Americans or their ancestors came from Europe, Africa, or Asia (the "Old World") within the past five centuries. Mainstream American culture is a Western culture largely derived from the traditions of European immigrants with influences from many other sources, such as traditions brought by slaves from Africa. More recent immigration from Asia and especially Latin America has added to a cultural mix that has been described as a homogenizing melting pot, and a heterogeneous salad bowl, with immigrants contributing to, and often assimilating into, mainstream American culture. Under the First Amendment to the Constitution, the United States is considered to have the strongest protections of free speech of any country. Flag desecration, hate speech, blasphemy, and lese majesty are all forms of protected expression. A 2016 Pew Research Center poll found that Americans were the most supportive of free expression of any polity measured. Additionally, they are the "most supportive of freedom of the press and the right to use the Internet without government censorship". The U.S. is a socially progressive country with permissive attitudes surrounding human sexuality. LGBTQ rights in the United States are among the most advanced by global standards. The American Dream, or the perception that Americans enjoy high levels of social mobility, plays a key role in attracting immigrants. Whether this perception is accurate has been a topic of debate. While mainstream culture holds that the United States is a classless society, scholars identify significant differences between the country's social classes, affecting socialization, language, and values. Americans tend to greatly value socioeconomic achievement, but being ordinary or average is promoted by some as a noble condition as well. The National Foundation on the Arts and the Humanities is an agency of the United States federal government that was established in 1965 with the purpose to "develop and promote a broadly conceived national policy of support for the humanities and the arts in the United States, and for institutions which preserve the cultural heritage of the United States." 

Literature 

Colonial American authors were influenced by John Locke and other Enlightenment philosophers. The American Revolutionary Period (1765–1783) is notable for the political writings of Benjamin Franklin, Alexander Hamilton, Thomas Paine, and Thomas Jefferson. Shortly before and after the Revolutionary War, the newspaper rose to prominence, filling a demand for anti-British national literature. An early novel is William Hill Brown's The Power of Sympathy, published in 1791. Writer and critic John Neal in the early- to mid-19th century helped advance America toward a unique literature and culture by criticizing predecessors such as Washington Irving for imitating their British counterparts, and by influencing writers such as Edgar Allan Poe, who took American poetry and short fiction in new directions. Ralph Waldo Emerson and Margaret Fuller pioneered the influential Transcendentalism movement; Henry David Thoreau, author of Walden, was influenced by this movement. The conflict surrounding abolitionism inspired writers, like Harriet Beecher Stowe, and authors of slave narratives, such as Frederick Douglass. Nathaniel Hawthorne's The Scarlet Letter (1850) explored the dark side of American history, as did Herman Melville's Moby-Dick (1851). Major American poets of the 19th century American Renaissance include Walt Whitman, Melville, and Emily Dickinson. Mark Twain was the first major American writer to be born in the West. Henry James achieved international recognition with novels like The Portrait of a Lady (1881). As literacy rates rose, periodicals published more stories centered around industrial workers, women, and the rural poor. Naturalism, regionalism, and realism were the major literary movements of the period. While modernism generally took on an international character, modernist authors working within the United States more often rooted their work in specific regions, peoples, and cultures. Following the Great Migration to northern cities, African-American and black West Indian authors of the Harlem Renaissance developed an independent tradition of literature that rebuked a history of inequality and celebrated black culture. An important cultural export during the Jazz Age, these writings were a key influence on Négritude, a philosophy emerging in the 1930s among francophone writers of the African diaspora. In the 1950s, an ideal of homogeneity led many authors to attempt to write the Great American Novel, while the Beat Generation rejected this conformity, using styles that elevated the impact of the spoken word over mechanics to describe drug use, sexuality, and the failings of society. Contemporary literature is more pluralistic than in previous eras, with the closest thing to a unifying feature being a trend toward self-conscious experiments with language. Twelve American laureates have won the Nobel Prize in Literature. 

Mass media 

The four major broadcasters in the U.S. are the National Broadcasting Company (NBC), Columbia Broadcasting System (CBS), American Broadcasting Company (ABC), and Fox Broadcasting Company (Fox). The four major broadcast television networks are all commercial entities. The Public Broadcasting Service (PBS) is the country's major non-commercial public broadcast network; it also provides educational programming through local PBS stations. The U.S. cable television system offers hundreds of channels catering to a variety of niches. In 2021, about 83% of Americans over age 12 listened to broadcast radio, while about 40% listened to podcasts. In the prior year, there were 15,460 licensed full-power radio stations in the U.S. according to the Federal Communications Commission (FCC). Public radio broadcasting is largely supplied by National Public Radio (NPR), incorporated in February 1970 under the Public Broadcasting Act of 1967. U.S. newspapers with a global reach and reputation include The Wall Street Journal, The New York Times, The Washington Post, and USA Today. About 800 publications are produced in Spanish. With few exceptions, newspapers are privately owned, either by large chains such as Gannett or McClatchy, which own dozens or even hundreds of newspapers; by small chains that own a handful of papers; or, in an increasingly rare situation, by individuals or families. Major cities often have alternative newspapers to complement the mainstream daily papers, such as The Village Voice in New York City and LA Weekly in Los Angeles. The five most-visited websites in the world are Google, YouTube, Facebook, Instagram, and ChatGPT—all of them American-owned. Other popular platforms used include X (formerly Twitter) and Amazon. In 2025, the U.S. was the world's second-largest video game market by revenue (after China). In 2015, the U.S. video game industry consisted of 2,457 companies that employed around 220,000 jobs and generated $30.4 billion in revenue. There are 444 game publishers, developers, and hardware companies in California alone. According to the Game Developers Conference (GDC), the U.S. is the top location for video game development, with 58% of the world's game developers based there in 2025. Media freedom was classified as "problematic" by Reporters Without Borders in 2026, and scholars have noted a significant rise in censorship and self-censorship in recent years. 

Theater 

The United States is well known for its theater. Mainstream theater in the United States derives from the old European theatrical tradition and has been heavily influenced by the British theater. By the middle of the 19th century, America had created new distinct dramatic forms in the Tom Shows, the showboat theater and the minstrel show. The central hub of the American theater scene is the Theater District in Manhattan, with its divisions of Broadway, off-Broadway, and off-off-Broadway. Many movie and television celebrities have gotten their big break working in New York productions. Outside New York City, many cities have professional regional or resident theater companies that produce their own seasons. The biggest-budget theatrical productions are musicals. U.S. theater has an active community theater culture. The Tony Awards recognize excellence in live Broadway theater and are presented at an annual ceremony in Manhattan. The awards are given for Broadway productions and performances. One is also given for regional theater. 

Visual arts 

Folk art in colonial America grew out of artisanal craftsmanship in communities that allowed commonly trained people to individually express themselves. It was distinct from Europe's tradition of high art, which was less accessible and generally less relevant to early American settlers. Cultural movements in art and craftsmanship in colonial America generally lagged behind those of Western Europe. For example, the prevailing medieval style of woodworking and primitive sculpture became integral to early American folk art, despite the emergence of Renaissance styles in England in the late 16th and early 17th centuries. The new English styles would have been early enough to make a considerable impact on American folk art, but American styles and forms had already been firmly adopted. Not only did styles change slowly in early America, but there was a tendency for rural artisans there to continue their traditional forms longer than their urban counterparts did—and far longer than those in Western Europe. The Hudson River School was a mid-19th-century movement in the visual arts tradition of European naturalism. The 1913 Armory Show in New York City, an exhibition of European modernist art, shocked the public and transformed the U.S. art scene. American Realism and American Regionalism sought to reflect and give America new ways of looking at itself. Georgia O'Keeffe, Marsden Hartley, and others experimented with new and individualistic styles, which would become known as American modernism. Major artistic movements such as the abstract expressionism of Jackson Pollock and Willem de Kooning and the pop art of Andy Warhol and Roy Lichtenstein developed largely in the United States. Major photographers include Alfred Stieglitz, Edward Steichen, Dorothea Lange, Edward Weston, James Van Der Zee, Ansel Adams, and Gordon Parks. The tide of modernism and then postmodernism has brought global fame to American architects, including Frank Lloyd Wright, Philip Johnson, and Frank Gehry. The Metropolitan Museum of Art in Manhattan is the largest art museum in the United States and the fourth-largest in the world. 

Music 

American folk music encompasses numerous music genres, variously known as traditional music, traditional folk music, contemporary folk music, or roots music. Many traditional songs have been sung within the same family or folk group for generations, and sometimes trace back to such origins as the British Isles, mainland Europe, or Africa. The rhythmic and lyrical styles of African-American music in particular have influenced American music. Banjos were brought to America through the slave trade. Minstrel shows incorporating the instrument into their acts led to its increased popularity and widespread production in the 19th century. The electric guitar, first invented in the 1930s, and mass-produced by the 1940s, had an enormous influence on popular music, in particular due to the development of rock and roll. The synthesizer, turntablism, and electronic music were also largely developed in the U.S. Elements from folk idioms such as the blues and old-time music were adopted and transformed into popular genres with global audiences. Jazz grew from blues and ragtime in the early 20th century, developing from the innovations and recordings of composers such as W.C. Handy and Jelly Roll Morton. Louis Armstrong and Duke Ellington increased its popularity early in the 20th century. Country music developed in the 1920s, bluegrass and rhythm and blues in the 1940s, and rock and roll in the 1950s. In the 1960s, Bob Dylan emerged from the folk revival to become one of the country's most celebrated songwriters. The musical forms of punk and hip hop both originated in the United States in the 1970s. The United States has the world's largest music market, with a total retail value of $15.9 billion in 2022, and is the largest exporter of music. Most of the world's major record companies are based in the U.S.; they are represented by the Recording Industry Association of America (RIAA). Mid-20th-century American pop stars, such as Frank Sinatra and Elvis Presley, became global celebrities and best-selling music artists, as have artists of the late 20th century, such as Michael Jackson, Madonna, Whitney Houston, and Mariah Carey, and of the early 21st century, such as Eminem, Britney Spears, Lady Gaga, Katy Perry, Taylor Swift and Beyoncé. 

Fashion 

The United States has the world's largest apparel market by revenue. Apart from professional business attire, American fashion is eclectic and predominantly informal. Americans' diverse cultural roots are reflected in their clothing; however, sneakers, jeans, T-shirts, and baseball caps are emblematic of American styles. New York, with its Fashion Week, is considered to be one of the "Big Four" global fashion capitals, along with Paris, Milan, and London. A study demonstrated that general proximity to Manhattan's Garment District has been synonymous with American fashion since its inception in the early 20th century. A number of well-known designer labels, among them Tommy Hilfiger, Ralph Lauren, Tom Ford and Calvin Klein, are headquartered in Manhattan. Labels cater to niche markets, such as preteens. New York Fashion Week is one of the most influential fashion shows in the world, and is held twice each year in Manhattan; the annual Met Gala, also in Manhattan, has been called the fashion world's "biggest night". 

Cinema 

The U.S. film industry has a worldwide influence and following. Hollywood, a district in central Los Angeles, the nation's second-most populous city, is also metonymous for the American filmmaking industry. The major film studios of the United States are the primary source of the most commercially successful movies selling the most tickets in the world. Largely centered in the New York City region from its beginnings in the late 19th century through the first decades of the 20th century, the U.S. film industry has since been primarily based in and around Hollywood. Nonetheless, American film companies have been subject to the forces of globalization in the 21st century, and an increasing number of films are made elsewhere. The Academy Awards, popularly known as "the Oscars", have been held annually by the Academy of Motion Picture Arts and Sciences since 1929, and the Golden Globes have been held annually since January 1944. The industry peaked in what is commonly referred to as the "Golden Age of Hollywood", from the early sound period until the early 1960s, with screen actors such as John Wayne and Marilyn Monroe becoming iconic figures. In the 1970s, "New Hollywood", or the "Hollywood Renaissance", was defined by grittier films influenced by French and Italian realist pictures of the post-war period. The 21st century has been marked by the rise of American streaming platforms, which came to rival traditional cinema. 

Cuisine 

Early settlers were introduced by Native Americans to foods such as turkey, sweet potatoes, corn, squash, and maple syrup. Of the most enduring and pervasive examples are variations of the native dish called succotash. Early settlers and later immigrants combined these with foods they were familiar with, such as wheat flour, beef, and milk, to create a distinctive American cuisine. New World crops, especially pumpkin, corn, potatoes, and turkey as the main course are part of a shared national menu on Thanksgiving, when many Americans prepare or purchase traditional dishes to celebrate the occasion. Characteristic American dishes such as apple pie, fried chicken, doughnuts, french fries, macaroni and cheese, ice cream, hamburgers, hot dogs, and American pizza derive from the recipes of various immigrant groups. Mexican dishes such as burritos and tacos preexisted the United States in areas later annexed from Mexico, and adaptations of Chinese cuisine as well as pasta dishes freely adapted from Italian sources are all widely consumed. American chefs have had a significant impact on society both domestically and internationally. In 1946, the Culinary Institute of America was founded by Katharine Angell and Frances Roth. This would become the United States' most prestigious culinary school, where many of the most talented American chefs would study prior to successful careers. The United States restaurant industry was projected at $899 billion in sales for 2020, and employed more than 15 million people, representing 10% of the nation's workforce directly. It is the country's second-largest private employer and the third-largest employer overall. The United States is home to over 220 Michelin-starred restaurants, 14 of which were awarded three stars. Wine has been produced in what is now the United States since the 1500s, with the first widespread production beginning in what is now New Mexico in 1628. In the modern U.S., wine production is undertaken in all fifty states, with California producing 84 percent of all U.S. wine. With more than 1,100,000 acres (4,500 km2) under vine, the United States is the fourth-largest wine-producing country in the world, after Italy, Spain, and France. The classic American diner, a casual restaurant type originally intended for the working class, emerged during the 19th century from converted railroad dining cars made stationary. The diner soon evolved into purpose-built structures whose number expanded greatly in the 20th century. The American fast-food industry developed alongside the nation's car culture. American restaurants developed the drive-in format in the 1920s, which they began to replace with the drive-through format by the 1940s. American fast-food restaurant chains, such as McDonald's, Burger King, Chick-fil-A, Kentucky Fried Chicken, Dunkin' Donuts and many others, have numerous outlets around the world. 

Sports 

The most popular spectator sports in the U.S. are American football, basketball, baseball, soccer, and ice hockey. Their premier leagues are, respectively, the National Football League, National Basketball Association, Major League Baseball, Major League Soccer, and the National Hockey League. All these leagues, excluding soccer, are considered to be preeminent in their respective sports worldwide. While most major U.S. sports such as baseball and American football have evolved out of European practices, basketball, volleyball, skateboarding, and snowboarding are American inventions, many of which have become popular worldwide. Lacrosse and surfing arose from Native American and Native Hawaiian activities that predate European contact. The U.S. professional sports market was approximately $69 billion in July 2013, roughly 50% larger than that of Europe, the Middle East, and Africa combined. Professional wrestling was widely popularized in the country, with the United States serving as the home country for World Wrestling Entertainment and All Elite Wrestling, the two largest wrestling promotions in the world. American football is by several measures the most popular spectator sport in the United States. Though American football does not have a substantial following in other nations, the NFL has the highest average attendance (67,254) and highest value ($23 billion in 2024) of any professional sports league in the world. Baseball has been regarded as the U.S. "national sport" since the late 19th century. The most-watched individual sports in the U.S. are golf and auto racing, particularly NASCAR and IndyCar. On the collegiate level, earnings for the member institutions exceed $1 billion annually, and college football and basketball attract large audiences, as the NCAA March Madness tournament and the College Football Playoff are some of the most watched national sporting events. In the U.S., the intercollegiate sports level serves as the main feeder system for professional and Olympic sports, with significant exceptions such as Minor League Baseball. This differs greatly from practices in nearly all other countries, where publicly and privately funded sports organizations serve this function. Eight Olympic Games have taken place in the United States, beginning with the 1904 Summer Olympics in St. Louis, Missouri. The U.S. is scheduled to host the 2028 Summer Olympics in Los Angeles and the 2034 Winter Olympics in Salt Lake City. U.S. athletes have won a total of 2,968 medals (1,179 gold) at the Olympic Games, the most of any country. In other international competition, the United States is the home of a number of prestigious events, including the America's Cup, World Baseball Classic, the U.S. Open, and the Masters Tournament. The U.S. men's national soccer team has qualified for eleven World Cups, while the women's national team has won the FIFA Women's World Cup and Olympic soccer tournament four and five times, respectively. The 1999 FIFA Women's World Cup was hosted by the United States. Its final match was attended by 90,185, setting the world record for largest women's sporting event crowd at the time. The United States hosted the 1994 FIFA World Cup and co-hosted, along with Canada and Mexico, the 2026 FIFA World Cup. 

See also 

Lists of U.S. state topics Outline of the United States 

Notes 

References 

Sources 

This article incorporates text from a free content work. Licensed under CC BY-SA IGO 3.0 (license statement/permission). Text taken from World Food and Agriculture – Statistical Yearbook 2023 , FAO, FAO.  

External links 

Key Development Forecasts for the United States from International Futures United States at the Encyclopædia Britannica 

Government Official U.S. Government web portal – gateway to government sites House – official website of the United States House of Representatives Senate – official website of the United States Senate White House – official website of the president of the United States Supreme Court – official website of the Supreme Court of the United States 

History Historical Documents – website from the National Center for Public Policy Research Historical Statistics – links to U.S. historical data 

Maps National Atlas of the United States – official maps from the U.S. Department of the Interior Wikimedia Atlas of the United States Geographic data related to United States at OpenStreetMap "Measure of America" – a variety of mapped information relating to health, education, income, safety and demographics in the United States 

[Human] Humans (Homo sapiens, meaning 'thinking man' or 'wise man') are the most abundant and widespread species of primates, characterized by bipedalism, minimal body hair, and large, complex brains enabling the development of advanced technology, culture, and language. Humans are highly social beings and tend to live in complex social structures composed of many cooperating and competing groups, from families and kinship networks to political states. Social interactions between humans have established a wide variety of values, social norms, and rituals, which bolster human society. Curiosity and the human desire to understand and influence the environment have motivated humanity's development of science, philosophy, religion, mythology and other fields of knowledge. Humans have a large and highly developed prefrontal cortex, the region of the brain associated with higher cognition. They are intelligent beings, capable of episodic memory, flexible facial expressions, self-awareness and a theory of mind. The human mind is capable of introspection (meta-cognition), private thought, imagination, volition and forming views on existence. Humans can also mentally travel through time (chronesthesia) which signifies episodic foresight. These attributes have allowed them to achieve large technological advancements through reason and the transmission of knowledge to future generations. In cultural anthropology, the cumulative preservation and advancement of knowledge across generations is known as the ratchet effect. Humans are omnivorous, capable of consuming a wide variety of plant and animal material, and have used fire to prepare and cook food. They can survive for up to eight weeks without food, and three or four days without water. Humans are generally diurnal, sleeping on average seven to nine hours per day. Human childbirth is dangerous, with a high risk of complications and death. Both the mother and the father typically provide care for human offspring, who are helpless at birth. Genes and the environment influence human biological variation in appearance, physiology, immune system, mental abilities, body size and lifespan. Though humans vary in many traits, any two humans are on average over 99% genetically similar, with the most genetically diverse populations from Africa. The greatest degree of genetic variation exists between males and females. On average, males have greater body strength and females generally have a higher body fat percentage. Females undergo menopause and become infertile potentially decades before the end of their lives. They also have a longer life span in almost every population around the world. The division into male and female gender roles has varied historically, and challenges to predominant gender norms have recurred in many societies. Humans evolved from other hominins in Africa several million years ago. Although some scientists equate humans with all members of the genus Homo, in common usage it generally refers to Homo sapiens, the only extant member. Homo sapiens emerged around 300,000 years ago and migrated out of Africa, gradually replacing local populations of archaic humans. Early humans were hunter-gatherers, before the invention of agriculture and domestication of animals led to permanent settlement in the Fertile Crescent and other cradles of civilization. As food surpluses enabled populations to become larger and denser, forms of governance developed, the use of writing became necessary, and a number of civilizations rose and fell. Humans have continued to expand, with over 8.3 billion humans occupying almost all regions of Earth in 2026, and have even visited the moon, the first known species to do so. This expansion, combined with industrialization, has led to environmental destruction and pollution that significantly contributes to the ongoing mass extinction of many other forms of life. 

Etymology and definition 

All modern humans are classified into the species Homo sapiens, coined by Carl Linnaeus in his 1735 work Systema Naturae. The generic name Homo is a learned 18th-century derivation from Latin homō, which refers to humans of either sex. The word human can refer to all members of the Homo genus. The name Homo sapiens means 'wise man' or 'knowledgeable man'. There is disagreement if certain extinct members of the genus, namely Neanderthals, should be included as a separate species of humans or as a subspecies of H. sapiens. Human is a loanword of Middle English from Old French humain, ultimately from Latin hūmānus, the adjectival form of homō ('man' – in the sense of humanity). The native English term man can refer to the species generally (a synonym for humanity) as well as to human males. It may also refer to individuals of either sex. Although the word animal is colloquially used as an antonym for human, and contrary to a common biological misconception, humans are animals. The word person is often used interchangeably with human, but philosophical debate exists as to whether personhood applies to all humans or all sentient beings, and further if a human can lose personhood (such as by going into a persistent vegetative state) and what the beginning of human personhood is. 

Evolution 

Humans belong to the biological family of great apes (family Hominidae, superfamily Hominoidea). The lineage of apes that eventually gave rise to humans first split from gibbons (family Hylobatidae), next orangutans (genus Pongo), then gorillas (genus Gorilla), and finally, chimpanzees and bonobos (genus Pan). The last split, between the human and chimpanzee–bonobo lineages, took place around 8–4 million years ago, in the late Miocene epoch. During this split, chromosome 2 was formed in humans from the joining of two other chromosomes, leaving humans with only 23 pairs of chromosomes, compared to 24 for the other apes. Following their split with chimpanzees and bonobos, the hominins diversified into many species and at least two distinct genera. All but one of these lineages – representing the genus Homo and its sole extant species Homo sapiens – are now extinct. 

The genus Homo evolved from Australopithecus. Though fossils from the transition are scarce, the earliest members of Homo share several key traits with Australopithecus. Due to the scant available evidence, the time of the divergence to the genus Homo does not have a consensus. Some studies using molecular clock techniques estimate the Homo genus appeared 4.30–2.56 million years ago, while others contest that some early Homo species are incorrectly included in the genus and therefore put this estimate at about 1.87 million years ago. The earliest record of Homo is the 2.8 million-year-old specimen LD 350-1 from Ethiopia, and the earliest named species are Homo habilis and Homo rudolfensis which evolved by 2.3 million years ago. H. erectus (the African variant is sometimes called H. ergaster) evolved 2 million years ago and was the first archaic human species to leave Africa and disperse across Eurasia. H. erectus also was the first to evolve a characteristically human body plan. Homo sapiens emerged in Africa at least 300,000 years ago from a species commonly designated as either H. heidelbergensis or H. rhodesiensis, the descendants of H. erectus that remained in Africa. H. sapiens migrated out of the continent, gradually replacing or interbreeding with local populations of archaic humans. Humans began exhibiting behavioral modernity about 160,000–70,000 years ago, and possibly earlier. This development was likely selected amidst natural climate change in Middle to Late Pleistocene Africa. The "out of Africa" migration took place in at least two waves, the first around 130,000 to 100,000 years ago, the second (Southern Dispersal) around 70,000 to 50,000 years ago. H. sapiens proceeded to colonize all the continents and larger islands except Antarctica, arriving in Eurasia 125,000 years ago, Australia around 65,000 years ago, the Americas around 15,000 years ago, and remote islands such as Hawaii, Easter Island, Madagascar, and New Zealand in the years 300 to 1280 CE. Human evolution was not a simple linear or branched progression but involved interbreeding between related species. Genomic research has shown that hybridization between substantially diverged lineages was common in human evolution. DNA evidence suggests that several genes of Neanderthal origin are present among all non sub-Saharan-African populations, and Neanderthals and other hominins, such as Denisovans, may have contributed up to 6% of their genome to present-day non sub-Saharan-African humans. Human evolution is characterized by a number of morphological, developmental, physiological, and behavioral changes that have taken place since the split between the last common ancestor of humans and chimpanzees. The most significant of these adaptations are hairlessness, obligate bipedalism, increased brain size and decreased sexual dimorphism (neoteny). The relationship between all these changes is the subject of ongoing debate. 

History 

Prehistory 

Until about 12,000 years ago, all humans lived as hunter-gatherers. The Neolithic Revolution (the invention of agriculture) first took place in Southwest Asia and spread through large parts of the Old World over the following millennia. It also occurred independently in Mesoamerica (about 6,000 years ago), China, Papua New Guinea, and the Sahel and West Savanna regions of Africa. The formation of permanent human settlements, the domestication of animals and the use of metal tools was followed by permanent food surplus, for the first time in history. Agriculture and sedentary lifestyle led to the emergence of early civilizations. 

Ancient 

An urban revolution took place in the 4th millennium BCE with the development of city-states, particularly Sumerian cities located in Mesopotamia. It was in these cities that the earliest known form of writing, cuneiform script, appeared around 3000 BCE. Other major civilizations to develop around this time were Ancient Egypt and the Indus Valley Civilization. They eventually traded with each other and invented technology such as wheels, plows and sails. Emerging by 3000 BCE, the Caral–Supe civilization is the oldest complex civilization in the Americas. Astronomy and mathematics were also developed and the Great Pyramid of Giza was built. There is evidence of a severe drought lasting about a hundred years that may have caused the decline of these civilizations, with new ones appearing in the aftermath. Babylonians came to dominate Mesopotamia while others, such as the Poverty Point culture, Minoans and the Shang dynasty, rose to prominence in new areas. The Late Bronze Age collapse around 1200 BCE resulted in the disappearance of a number of civilizations and the beginning of the Greek Dark Ages. During this period iron started replacing bronze, leading to the Iron Age. In the 5th century BCE, history started being recorded as a discipline, which provided a much clearer picture of life at the time. Between the 8th and 6th century BCE, Europe entered the classical antiquity age, a period when ancient Greece and ancient Rome flourished. Around this time other civilizations also came to prominence. The Maya civilization started to build cities and create complex calendars. In Africa, the Kingdom of Aksum overtook the declining Kingdom of Kush and facilitated trade between India and the Mediterranean. In West Asia, the Achaemenid Empire's system of centralized governance became the precursor to many later empires, while the Gupta Empire in India and the Han dynasty in China have been described as golden ages in their respective regions. 

Post-classical 

Following the fall of the Western Roman Empire in 476, Europe entered the Middle Ages. During this period, Christianity and the Church would act as a source of authority and education. In the Middle East, Islam became the prominent religion and expanded into North Africa. It led to an Islamic Golden Age, inspiring achievements in architecture, the revival of old advances in science and technology, and the formation of a distinct way of life. The Christian and Islamic worlds would eventually clash, with the Kingdom of England, the Kingdom of France and the Holy Roman Empire declaring a series of holy wars to regain control of the Holy Land from Muslims. In the Americas, between 200 and 900 CE Mesoamerica was in its Classic Period, while further north, complex Mississippian societies would arise starting around 800 CE. The Mongol Empire would conquer much of Eurasia in the 13th and 14th centuries. Over this same time period, the Mali Empire in Africa grew to be the largest empire on the continent, stretching from Senegambia to Ivory Coast. Oceania would see the rise of the Tuʻi Tonga Empire which expanded across many islands in the South Pacific. By the late 15th century, the Aztecs and Inca had become the dominant power in Mesoamerica and the Andes, respectively. 

Modern 

The early modern period in Europe and the Near East (c. 1450–1800) began with the final defeat of the Byzantine Empire, and the rise of the Ottoman Empire. Meanwhile, Japan entered the Edo period, the Qing dynasty rose in China and the Mughal Empire ruled much of India. Europe underwent the Renaissance, starting in the 15th century, and the Age of Discovery began with the exploring and colonizing of new regions. This included the colonization of the Americas and the Columbian Exchange. This expansion led to the Atlantic slave trade and the genocide of the Americas' indigenous peoples. This period also marked the Scientific Revolution, with great advances in mathematics, mechanics, astronomy and physiology. 

The late modern period (1800–present) saw the Industrial and Technological Revolution bring such discoveries as imaging technology, major innovations in transport, and energy development. Influenced by Enlightenment ideals, the Americas and Europe experienced a period of political revolutions known as the Age of Revolution. The Napoleonic Wars raged through Europe in the early 1800s, Spain lost most of its colonies in the New World, while Europeans continued expansion into Africa – where European control went from 10% to almost 90% in less than 50 years – and Oceania. In the 19th century, the British Empire expanded to become the world's largest empire. A tenuous balance of power among European nations collapsed in 1914 with the outbreak of the First World War, one of the deadliest conflicts in history. In the 1930s, a worldwide economic crisis led to the rise of authoritarian regimes and a Second World War, involving almost all of the world's countries. The war's destruction led to the collapse of most global empires, leading to widespread decolonization. Following the conclusion of the Second World War in 1945, the United States and the Soviet Union emerged as the remaining global superpowers. This led to a Cold War that saw a struggle for global influence, including a nuclear arms race and a space race, ending in the collapse of the Soviet Union. The current Information Age, spurred by the development of the Internet and artificial intelligence systems, sees the world becoming increasingly globalized and interconnected. 

Habitat and population 

Early human settlements were dependent on proximity to water and – depending on the lifestyle – other natural resources used for subsistence, such as populations of animal prey for hunting and arable land for growing crops and grazing livestock. Modern humans, however, have a great capacity for altering their habitats by means of technology, irrigation, urban planning, construction, deforestation and desertification. Human settlements continue to be vulnerable to natural disasters, especially those placed in hazardous locations and with low quality of construction. Grouping and deliberate habitat alteration is often done with the goals of providing protection, accumulating comforts or material wealth, expanding the available food, improving aesthetics, increasing knowledge or enhancing the exchange of resources. Humans are one of the most adaptable species, despite having a low or narrow tolerance for many of the Earth's extreme environments. Currently the species is present in all eight biogeographical realms, although their presence in the Antarctic realm is limited to research stations and annually there is a population decline in the winter months of this realm. Humans have had a dramatic effect on the environment. Human population growth, industrialization, land development, overconsumption and combustion of fossil fuels have led to environmental destruction and pollution that significantly contributes to the ongoing mass extinction of other forms of life. Within the last century, humans have also explored the deep sea and outer space. Human habitation within these hostile environments is restrictive and expensive, typically limited in duration, and restricted to scientific, military, or industrial expeditions. Humans have visited the Moon and made their presence known on other celestial bodies through human-made robotic spacecraft. Since 2000, there has been continuous human presence in space through habitation on the International Space Station. By using advanced tools and clothing, humans have been able to extend their tolerance to a wide variety of temperatures, humidities, and altitudes. As a result, humans are a cosmopolitan species found in almost all regions of the world, including tropical rainforest, arid desert, extremely cold arctic regions, and heavily polluted cities; in comparison, most other species are confined to a few geographical areas by their limited adaptability. The human population is not, however, uniformly distributed on the Earth's surface, because the population density varies from one region to another, and large stretches of surface are almost completely uninhabited, like Antarctica and vast swathes of the ocean. Most humans (61%) live in Asia; the remainder live in the Americas (14%), Africa (14%), Europe (11%), and Oceania (0.5%). 

Estimates of the population at the time agriculture emerged in around 10,000 BC have ranged between 1 million and 15 million. Around 50–60 million people lived in the combined eastern and western Roman Empire in the 4th century AD. Bubonic plagues, first recorded in the 6th century AD, reduced the population by 50%, with the Black Death killing 75–200 million people in Eurasia and North Africa alone. Human population is believed to have reached one billion in 1800. It has since then increased exponentially, reaching two billion in 1930 and three billion in 1960, four in 1975, five in 1987 and six billion in 1999. It passed seven billion in 2011 and passed eight billion in 2022. It took over two million years of human prehistory and history for the human population to reach one billion and only 207 years more to grow to 7 billion. The combined biomass of the carbon of all the humans on Earth in 2018 was estimated at 60 million tons, about 10 times larger than that of all non-domesticated mammals. In 2018, 4.2 billion humans (55%) lived in urban areas, up from 751 million in 1950. The most urbanized regions are Northern America (82%), Latin America (81%), Europe (74%) and Oceania (68%), with Africa and Asia having nearly 90% of the world's 3.4 billion rural population. Problems for humans living in cities include various forms of pollution and crime, especially in inner city and suburban slums. 

Biology 

Anatomy and physiology 

Most aspects of human physiology are closely homologous to corresponding aspects of animal physiology. The dental formula of humans is: 2.1.2.32.1.2.3, like other catarrhines. Humans have proportionately shorter palates and much smaller teeth than other primates. They are the only primates to have short, relatively flush canine teeth. Humans have characteristically crowded teeth, with gaps from lost teeth usually closing up quickly in young individuals. Humans are gradually losing their third molars, with some individuals having them congenitally absent. Humans share with chimpanzees a vestigial tail, appendix, flexible shoulder joints, grasping fingers and opposable thumbs. Humans also have a more barrel-shaped chest in contrast to the funnel shape of other apes, an adaptation for bipedal respiration. Apart from bipedalism and brain size, humans differ from chimpanzees mostly in smelling, hearing and digesting proteins. While humans have a density of hair follicles comparable to other apes, it is predominantly vellus hair, most of which is so short and wispy as to be practically invisible. Humans have about 2 million sweat glands spread over their entire bodies, many more than chimpanzees, whose sweat glands are scarce and are mainly located on the palm of the hand and on the soles of the feet. It is estimated that the worldwide average height for an adult human male is about 171 cm (5 ft 7 in), while the worldwide average height for adult human females is about 159 cm (5 ft 3 in). Shrinkage of stature may begin in middle age in some individuals but tends to be typical in the extremely aged. Throughout history, human populations have universally become taller, probably as a consequence of better nutrition, healthcare, and living conditions. The average mass of an adult human is 59 kg (130 lb) for females and 77 kg (170 lb) for males. Like many other conditions, body weight and body type are influenced by both genetic susceptibility and environment and varies greatly among individuals. Humans have a far faster and more accurate throw than other animals. Humans are also among the best long-distance runners in the animal kingdom, but slower over short distances. Humans' thinner body hair and more productive sweat glands help avoid heat exhaustion while running for long distances. Compared to other apes, the human heart produces greater stroke volume and cardiac output and the aorta is proportionately larger. 

Genetics 

Humans are, like all animals, plants, and fungi, a eukaryotic, and like most animals a diploid species. Each somatic cell has two sets of 23 chromosomes, each set received from one parent; gametes have only one set of chromosomes, which is a mixture of the two parental sets. Among the 23 pairs of chromosomes, there are 22 pairs of autosomes and one pair of sex chromosomes. In humans, sex determination is primarily mediated by the SRY gene located on the Y chromosome. While the typical configuration follows an XY sex-determination system (XX for females, XY for males), the genetic expression of the Sex-determining Region Y protein is the critical factor in initiating male gonadal differentiation. Consequently, individuals may be born with a chromosomal genotype that does not align with their phenotypic sex, a condition often resulting in infertility due to the absence of specific genes required for gametogenesis. Genes and environment influence human biological variation in visible characteristics, physiology, disease susceptibility and mental abilities. The exact influence of genes and environment on certain traits is not well understood. While no humans – not even monozygotic twins – are genetically identical, two humans on average will have a genetic similarity of 99.5%-99.9%. This makes them more homogeneous than other great apes, including chimpanzees. This small variation in human DNA compared to many other species suggests a population bottleneck during the Late Pleistocene (around 100,000 years ago), in which the human population was reduced to a small number of breeding pairs. The forces of natural selection have continued to operate on human populations, with evidence that certain regions of the genome display directional selection in the past 15,000 years. In 1984, the US government began to plan the human genome project (HGP), which officially started in 1990. Utilizing the data from the HGP, 92% of the human genome was first sequenced in 2001. After the HGP was completed in April 2003, the All of Us Research Program in 2022 released its first major dataset which included nearly 100,000 whole genome sequences. By April 2023, approximately 245,000 genomes had been sequenced. In 2012 the International HapMap Project had compared the genomes of 1,184 individuals from 11 populations and identified 1.6 million single nucleotide polymorphisms. African populations harbor the highest number of private genetic variants. While many of the common variants found in populations outside of Africa are also found on the African continent, there are still large numbers that are private to these regions, especially Oceania and the Americas. By 2010 estimates, humans have approximately 22,000 genes. By comparing mitochondrial DNA, which is inherited only from the mother, geneticists have concluded that the last female common ancestor whose genetic marker is found in all modern humans, the so-called mitochondrial Eve, must have lived around 90,000 to 200,000 years ago. 

Life cycle 

Most human reproduction takes place by internal fertilization via sexual intercourse, but can also occur through assisted reproductive technology procedures. The average gestation period is 38 weeks, but a normal pregnancy can vary by up to 37 days. Embryonic development in the human covers the first eight weeks of development; at the beginning of the ninth week the embryo is termed a fetus. Humans are able to induce early labor or perform a caesarean section if the child needs to be born earlier for medical reasons. In developed countries, infants are typically 3–4 kg (7–9 lb) in weight and 47–53 cm (19–21 in) in height at birth. However, low birth weight is common in developing countries, and contributes to the high levels of infant mortality in these regions. Compared with other species, human childbirth is dangerous, with a much higher risk of complications and death. The size of the fetus's head is more closely matched to the pelvis than in other primates. The reason for this is not completely understood, but it contributes to a painful labor that can last 24 hours or more. The chances of a successful labor increased significantly during the 20th century in wealthier countries with the advent of new medical technologies. In contrast, pregnancy and natural childbirth remain hazardous ordeals in developing regions of the world, with maternal death rates approximately 100 times greater than in developed countries. Both the mother and the father provide care for human offspring, in contrast to other primates, where parental care is mostly done by the mother. Helpless at birth, humans continue to grow for some years, typically reaching sexual maturity at 15 to 17 years of age. The human life span has been split into various stages ranging from three to twelve. Common stages include infancy, childhood, adolescence, adulthood and old age. The lengths of these stages have varied across cultures and time periods but is typified by an unusually rapid growth spurt during adolescence. Human females undergo menopause and become infertile at around the age of 50. It has been proposed that menopause increases a woman's overall reproductive success by allowing her to invest more time and resources in her existing offspring, and in turn their children (the grandmother hypothesis), rather than by continuing to bear children into old age. The life span of an individual depends on two major factors, genetics and lifestyle choices. For various reasons, including biological/genetic causes, women live on average about four years longer than men. As of 2018, the global average life expectancy at birth of a girl is estimated to be 74.9 years compared to 70.4 for a boy. There are significant geographical variations in human life expectancy, mostly correlated with economic development – for example, life expectancy at birth in Hong Kong is 87.6 years for girls and 81.8 for boys, while in the Central African Republic, it is 55.0 years for girls and 50.6 for boys. The developed world is generally aging, with the median age around 40 years. In the developing world, the median age is between 15 and 20 years. While one in five Europeans is 60 years of age or older, only one in twenty Africans is 60 years of age or older. In 2012, the United Nations estimated that there were 316,600 living centenarians (humans of age 100 or older) worldwide. 

Diet 

Humans are omnivorous, opportunistic feeders capable of consuming a wide variety of plant and animal material. Human groups have adopted a range of diets from purely vegan to primarily carnivorous. In some cases, dietary restrictions in humans can lead to deficiency diseases; however, stable human groups have adapted to many dietary patterns through both genetic specialization and cultural conventions to use nutritionally balanced food sources. The human diet is prominently reflected in human culture and has led to the development of food science. Until the development of agriculture, Homo sapiens employed a hunter-gatherer method as their sole means of food collection. This involved combining stationary food sources (such as fruits, grains, tubers, and mushrooms, insect larvae and aquatic mollusks) with wild game, which must be hunted and captured in order to be consumed. It has been proposed that humans have used fire to prepare and cook food since the time of Homo erectus. Human domestication of wild plants began about 11,700 years ago, leading to the development of agriculture, a gradual process called the Neolithic Revolution. These dietary changes may also have altered human biology; the spread of dairy farming provided a new and rich source of food, leading to the evolution of the ability to digest lactose in some adults. The types of food consumed, and how they are prepared, have varied widely by time, location, and culture. In general, humans can survive for up to eight weeks without food, depending on stored body fat. Survival without water is usually limited to three or four days, with a maximum of one week. In 2020, it was estimated 9 million humans die every year from causes directly or indirectly related to starvation. Childhood malnutrition is also common and contributes to the global burden of disease. However, global food distribution is not even, and obesity among some human populations has increased rapidly, leading to health complications and increased mortality in some developed and a few developing countries. Worldwide, over one billion people are obese, while in the United States 35% of people are obese, leading to this being described as an "obesity epidemic." Obesity is caused by consuming more calories than are expended, so excessive weight gain is usually caused by an energy-dense diet. Food consumption is the first step of the digestive process, in which humans ultimately expel feces ranging in frequency from multiple times per day to multiple times per week. 

Biological variation 

There is biological variation in the human species – with traits such as blood type, genetic diseases, cranial features, facial features, organ systems, eye color, hair color and texture, height and build, and skin color varying across the globe. The typical height of an adult human is between 1.4 and 1.9 m (4 ft 7 in and 6 ft 3 in), although this varies significantly depending on sex, ethnic origin, and family bloodlines. Body size is partly determined by genes and is also significantly influenced by environmental factors such as diet, exercise, and sleep patterns. 

There is evidence that populations have adapted genetically to various external factors. The genes that allow adult humans to digest lactose are present in high frequencies in populations that have long histories of cattle domestication and are more dependent on cow milk. Sickle cell anemia, which may provide increased resistance to malaria, is frequent in populations where malaria is endemic. Populations that have for a very long time inhabited specific climates tend to have developed specific phenotypes that are beneficial for those environments – short stature and stocky build in cold regions, tall and lanky in hot regions, and with high lung capacities or other adaptations at high altitudes. Some populations have evolved highly unique adaptations to very specific environmental conditions, such as those advantageous to ocean-dwelling lifestyles and freediving in the Bajau. Human hair ranges in color from red to blond to brown to black, which is the most frequent. Hair color depends on the amount of melanin, with concentrations fading with increased age, leading to grey or even white hair. Skin color can range from darkest brown to lightest peach, or even nearly white or colorless in cases of albinism. It tends to vary clinally and generally correlates with the level of ultraviolet radiation in a particular geographic area, with darker skin mostly around the equator. Skin darkening may have evolved as protection against ultraviolet solar radiation. Light skin pigmentation protects against depletion of vitamin D, which requires sunlight to make. Human skin also has a capacity to darken (tan) in response to exposure to ultraviolet radiation. There is relatively little variation between human geographical populations, and most of the variation that occurs is at the individual level. Much of human variation is continuous, often with no clear points of demarcation. Genetic data shows that no matter how population groups are defined, two people from the same population group are almost as different from each other as two people from any two different population groups. Dark-skinned populations that are found in Africa, Australia, and South Asia are not closely related to each other. Genetic research has demonstrated that human populations native to the African continent are the most genetically diverse and genetic diversity decreases with migratory distance from Africa, possibly the result of bottlenecks during human migration. These non-African populations acquired new genetic inputs from local admixture with archaic populations and have much greater variation from Neanderthals and Denisovans than is found in Africa, though Neanderthal admixture into African populations may be underestimated. Furthermore, recent studies have found that populations in sub-Saharan Africa, and particularly West Africa, have ancestral genetic variation which predates modern humans and has been lost in most non-African populations. Some of this ancestry is thought to originate from admixture with an unknown archaic hominin that diverged before the split of Neanderthals and modern humans. Humans are a gonochoric species, meaning they are divided into male and female sexes. The greatest degree of genetic variation exists between males and females. While the nucleotide genetic variation of individuals of the same sex across global populations is no greater than 0.1%–0.5%, the genetic difference between males and females is between 1% and 2%. Males on average are 15% heavier and 15 cm (6 in) taller than females. On average, males have about 40–50% more upper-body strength and 20–30% more lower-body strength than females at the same weight, due to higher amounts of muscle and larger muscle fibers. Females generally have a higher body fat percentage than males. Females have lighter skin than males of the same population; this has been explained by a higher need for vitamin D in females during pregnancy and lactation. As there are chromosomal differences between females and males, some X and Y chromosome-related conditions and disorders only affect either males or females. After allowing for body weight and volume, the male voice is usually an octave deeper than the female voice. Females have a longer life span in almost every population around the world. There are intersex conditions in the human population, however these are rare. 

Psychology 

The human brain is the locus of "higher" order functioning such as thought, reasoning, and abstraction. These cognitive processes constitute the mind, and, along with their behavioral consequences, are studied in the field of psychology. Humans have a larger and more developed prefrontal cortex than other primates, the region of the brain associated with higher cognition. This has led humans to proclaim themselves to be more intelligent than any other known species. Objectively defining intelligence is difficult, with other animals adapting senses and excelling in areas that humans are unable to. There are some traits that, although not strictly unique, do set humans apart from other animals. Humans may be the only animals who have episodic memory and who can engage in "mental time travel". Even compared with other social animals, humans have an unusually high degree of flexibility in their facial expressions. Humans are the only animals known to cry emotional tears. Humans are one of the few animals able to self-recognize in mirror tests and there is also debate over to what extent humans are the only animals with a theory of mind. 

Sleep and dreaming 

Humans are generally diurnal. The average sleep requirement is between seven and nine hours per day for an adult and nine to ten hours per day for a child; elderly people usually sleep for six to seven hours. Having less sleep than this is common among humans, even though sleep deprivation can have negative health effects. A sustained restriction of adult sleep to four hours per day has been shown to correlate with changes in physiology and mental state, including reduced memory, fatigue, aggression, and bodily discomfort. During sleep humans dream, where they experience sensory images and sounds. Dreaming is stimulated by the pons and mostly occurs during the REM phase of sleep. The length of a dream can vary, from a few seconds up to 30 minutes. Humans have three to five dreams per night, and some may have up to seven. Dreamers are more likely to remember the dream if awakened during the REM phase. The events in dreams are generally outside the control of the dreamer, with the exception of lucid dreaming, where the dreamer is self-aware. Dreams can at times make a creative thought occur or give a sense of inspiration, as is the case with some notable works of fiction, scientific concepts, and art. 

Consciousness and thought 

Human consciousness, at its simplest, is sentience or awareness of internal or external existence. Despite centuries of analyses, definitions, explanations and debates by philosophers and scientists, the underlying nature of consciousness remains enigmatic and poorly understood, being "at once the most familiar and most mysterious aspect of our lives". The only widely agreed notion about the topic is the intuition that it exists. Opinions differ about what exactly needs to be studied and explained as consciousness. Some philosophers divide consciousness into phenomenal consciousness, which is sensory experience itself, and access consciousness, which can be used for reasoning or directly controlling actions. It is sometimes synonymous with 'the mind', and at other times, an aspect of it. Historically it is associated with introspection, private thought, imagination and volition. It now often includes some kind of experience, cognition, feeling or perception. It may be 'awareness', or 'awareness of awareness', or self-awareness. There might be different levels or orders of consciousness, or different kinds of consciousness, or just one kind with different features. The process of acquiring knowledge and understanding through thought, experience, and the senses is known as cognition. The human brain perceives the external world through the senses, and each individual human is influenced greatly by their experiences, leading to subjective views of existence and the passage of time. The nature of thought is central to psychology and related fields. Cognitive psychology studies cognition, the mental processes underlying behavior. Largely focusing on the development of the human mind through the life span, developmental psychology seeks to understand how people come to perceive, understand, and act within the world and how these processes change as they age. This may focus on intellectual, cognitive, neural, social, or moral development. Psychologists have developed intelligence tests and the concept of intelligence quotient in order to assess the relative intelligence of human beings and study its distribution among population. 

Motivation and emotion 

Human motivation is not yet wholly understood. From a psychological perspective, Maslow's hierarchy of needs is a well-established theory that can be defined as the process of satisfying certain needs in ascending order of complexity. From a more general, philosophical perspective, human motivation can be defined as a commitment to, or withdrawal from, various goals requiring the application of human ability. Furthermore, incentive and preference are both factors, as are any perceived links between incentives and preferences. Volition may also be involved, in which case willpower is also a factor. Ideally, both motivation and volition ensure the selection, striving for, and realization of goals in an optimal manner, a function beginning in childhood and continuing throughout a lifetime in a process known as socialization. Emotions are biological states associated with the nervous system brought on by neurophysiological changes variously associated with thoughts, feelings, behavioral responses, and a degree of pleasure or displeasure. They are often intertwined with mood, temperament, personality, disposition, creativity, and motivation. Emotion has a significant influence on human behavior and their ability to learn. Acting on extreme or uncontrolled emotions can lead to social disorder and crime, with studies showing criminals may have a lower emotional intelligence than normal. Emotional experiences perceived as pleasant, such as joy, interest or contentment, contrast with those perceived as unpleasant, like anxiety, sadness, anger, and despair. Happiness, or the state of being happy, is a human emotional condition. The definition of happiness is a common philosophical topic. Some define it as experiencing the feeling of positive emotional affects, while avoiding the negative ones. Others see it as an appraisal of life satisfaction or quality of life. Recent research suggests that being happy might involve experiencing some negative emotions when humans feel they are warranted. 

Sexuality and love 

For humans, sexuality involves biological, erotic, physical, emotional, social, or spiritual feelings and behaviors. Because it is a broad term, which has varied with historical contexts over time, it lacks a precise definition. The biological and physical aspects of sexuality largely concern the human reproductive functions, including the human sexual response cycle. Sexuality also affects and is affected by cultural, political, legal, philosophical, moral, ethical, and religious aspects of life. Sexual desire (libido) is a basic mental state present at the beginning of sexual behavior. Studies have found that men report higher sexual desire than women and masturbate more frequently. Humans can fall anywhere along a continuous scale of sexual orientation, although most humans are heterosexual. While homosexual behavior occurs in some other animals, only humans and domestic sheep have so far been found to exhibit exclusive preference for same-sex relationships. Most evidence supports nonsocial, biological causes of sexual orientation, as cultures that are very tolerant of homosexuality do not have significantly higher rates of it. Research in neuroscience and genetics suggests that other aspects of human sexuality are biologically influenced as well. Love most commonly refers to a feeling of strong attraction or emotional attachment. It can be impersonal (the love of an object, ideal, or strong political or spiritual connection) or interpersonal (love between humans). In the emotional state of romantic love, neurochemicals such as dopamine, norepinephrine, serotonin, and oxytocin stimulate the brain's reward system through the mesocorticolimbic pathway (MCL). This dopaminergic pathway may result in side effects such as tachycardia (increased heart rate), loss of appetite, insomnia (sleeplessness), and intense euphoria. 

Culture 

Humanity's large set of intellectual skills were a key factor in the species' eventual technological advancement and concomitant domination of the biosphere. Disregarding extinct hominids, humans are the only animals known to teach generalizable information, innately deploy recursive embedding to generate and communicate complex concepts, engage in the "folk physics" required for competent tool design, or cook food in the wild. The ability to preserve and teach information to future generations in one or more societies is known as the ratchet effect in cultural anthropology. According to the metaphor, once a group of people understand a technology, it is hard for them to unlearn it, and may be used to learn technologies of further complexity. Teaching and learning preserves the cultural and ethnographic identity of human societies. Other traits and behaviors that are mostly unique to humans include starting fires, phoneme structuring and vocal learning. 

Language 

While many species communicate, language is unique to humans, a defining feature of humanity, and a cultural universal. Unlike the limited communication-systems of other animals, human language is open – an infinite number of meanings can be produced by combining a limited number of symbols. Human language also has the capacity of displacement, using words to represent things and happenings that are not presently or locally occurring but reside in the shared imagination of interlocutors. Language differs from other forms of communication in that it is modality independent; people can convey the same meanings through different media: audibly in speech, visually by sign language or writing, and through tactile media such as braille. Language is central to communication between humans, and to the sense of identity that can unite nations, cultures and ethnic groups. As of 1996 humans used approximately six thousand different languages, including sign languages, and many thousands more are extinct. 

The arts 

Human arts can take many forms including visual, literary, and performing. Visual art can range from paintings and sculptures to film, fashion design, and architecture. Literary arts can include prose, poetry, and dramas. The performing arts generally involve theater, music, and dance. Humans often combine the different forms (for example, music videos). Other entities that have been described as having artistic qualities include food preparation, video games, and medicine. As well as providing entertainment and transferring knowledge, the arts are also used for political purposes. Art is a defining characteristic of humans and there is evidence for a relationship between creativity and language. The earliest evidence of art was shell engravings made by Homo erectus 300,000 years before modern humans evolved. Art attributed to H. sapiens existed at least 75,000 years ago, with jewelry and drawings found in caves in South Africa. There are various hypotheses as to why humans have adapted to the arts. These include allowing them to better problem solve issues, providing a means to control or influence other humans, encouraging cooperation and contribution within a society or increasing the chance of attracting a potential mate. The use of imagination developed through art, combined with logic may have given early humans an evolutionary advantage. Evidence of humans engaging in musical activities predates cave art and so far music has been practiced by virtually all known human cultures. There exists a wide variety of music genres and ethnic musics; with humans' musical abilities being related to other abilities, including complex social human behaviors. It has been shown that human brains respond to music by becoming synchronized with the rhythm and beat, a process called entrainment. Dance is also a form of human expression found in all cultures and may have evolved as a way to help early humans communicate. Listening to music and observing dance stimulates the orbitofrontal cortex and other pleasure sensing areas of the brain. Unlike speaking, reading and writing does not come naturally to humans and must be taught. Still, literature has been present before the invention of words and language, with 30,000-year-old paintings on walls inside some caves portraying a series of dramatic scenes. One of the oldest surviving works of literature is the Epic of Gilgamesh, first engraved on ancient Babylonian tablets about 4,000 years ago. Beyond simply passing down knowledge, the use and sharing of imaginative fiction through stories might have helped develop humans' capabilities for communication and increased the likelihood of securing a mate. Storytelling may also be used as a way to provide the audience with moral lessons and encourage cooperation. 

Tools and technologies 

Stone tools were used by proto-humans at least 2.5 million years ago. The use and manufacture of tools has been put forward as the ability that defines humans more than anything else and has historically been seen as an important evolutionary step. The technology became much more sophisticated about 1.8 million years ago, with the controlled use of fire beginning around 1 million years ago. The wheel and wheeled vehicles appeared simultaneously in several regions some time in the fourth millennium BC. The development of more complex tools and technologies allowed land to be cultivated and animals to be domesticated, thus proving essential in the development of agriculture – what is known as the Neolithic Revolution. China developed paper, the printing press, gunpowder, the compass and other important inventions. The continued improvements in smelting allowed forging of copper, bronze, iron and eventually steel, which is used in railways, skyscrapers and many other products. This coincided with the Industrial Revolution, where the invention of automated machines brought major changes to humans' lifestyles. Modern technology is observed as progressing exponentially, with major innovations in the 20th century including: electricity, penicillin, semiconductors, internal combustion engines, the Internet, nitrogen fixing fertilizers, airplanes, computers, automobiles, contraceptive pills, nuclear fission, the green revolution, radio, scientific plant breeding, rockets, air conditioning, television and the assembly line. 

Religion and spirituality 

Definitions of religion vary; according to one definition, a religion is a belief system concerning the supernatural, sacred or divine, and practices, values, institutions and rituals associated with such belief. Some religions also have a moral code. The evolution and the history of the first religions have become areas of active scientific investigation. Credible evidence of religious behavior dates to the Middle Paleolithic era (45–200 thousand years ago). It may have evolved to play a role in helping enforce and encourage cooperation between humans. Religion manifests in diverse forms. Religion can include a belief in life after death, the origin of life, the nature of the universe (religious cosmology) and its ultimate fate (eschatology), and moral or ethical teachings. Views on transcendence and immanence vary substantially; traditions variously espouse monism, deism, pantheism, and theism (including polytheism and monotheism). Although measuring religiosity is difficult, a majority of humans profess some variety of religious or spiritual belief. In 2015 the plurality were Christian followed by Muslims, Hindus and Buddhists. As of 2015, about 16%, or slightly under 1.2 billion humans, were irreligious, including those with no religious beliefs or no identity with any religion. 

Science and philosophy 

A defining attribute of humanity is the capacity for cumulative cultural evolution by transmitting knowledge across generations by continuously building on past knowledge to develop tools, scientific laws and other advancements to pass on to future generations. This accumulated knowledge can be tested to answer various questions or make predictions about how the universe functions, driving ongoing human advancement. This phenomenon is known as the ratchet effect in cultural anthropology. Aristotle has been described as the first scientist, and preceded the rise of scientific thought through the Hellenistic period. Other early advances in science came from the Han dynasty in China and during the Islamic Golden Age. The scientific revolution, near the end of the Renaissance, led to the emergence of modern science. A chain of events and influences led to the development of the scientific method, a process of observation and experimentation that is used to differentiate science from pseudoscience. An understanding of mathematics is unique to humans, although other species of animals have some numerical cognition. All of science can be divided into three major branches, the formal sciences (e.g., logic and mathematics), which are concerned with formal systems, the applied sciences (e.g., engineering, medicine), which are focused on practical applications, and the empirical sciences, which are based on empirical observation and are in turn divided into natural sciences (e.g., physics, chemistry, biology) and social sciences (e.g., psychology, economics, sociology). Philosophy is a field of study where humans seek to understand fundamental truths about themselves and the world in which they live. Philosophical inquiry has been a major feature in the development of humans' intellectual history. It has been described as the "no man's land" between definitive scientific knowledge and dogmatic religious teachings. Major fields of philosophy include metaphysics, epistemology, logic, and axiology (which includes ethics and aesthetics). 

Society 

Society is the system of organizations and institutions arising from interaction between humans. Humans are highly social and tend to live in large complex social groups. They can be divided into different groups according to their income, wealth, power, reputation and other factors. The structure of social stratification and the degree of social mobility differs, especially between modern and traditional societies. Human groups range from the size of families to nations. The first form of human social organization is thought to have resembled hunter-gatherer band societies. 

Gender 

Human societies typically exhibit gender identities and gender roles that distinguish between masculine and feminine characteristics and prescribe the range of acceptable behaviors and attitudes for their members based on their sex. The most common categorization is a gender binary of men and women. Some societies recognize a third gender, or less commonly a fourth or fifth. In some other societies, non-binary is used as an umbrella term for a range of gender identities that are not solely male or female. Gender roles are often associated with a division of norms, practices, dress, behavior, rights, duties, privileges, status, and power, with men enjoying more rights and privileges than women in most societies, both today and in the past. As a social construct, gender roles are not fixed and vary historically within a society. Challenges to predominant gender norms have recurred in many societies. Little is known about gender roles in the earliest human societies. Early modern humans probably had a range of gender roles similar to that of modern cultures from at least the Upper Paleolithic, while the Neanderthals were less sexually dimorphic and there is evidence that the behavioral difference between males and females was minimal. 

Kinship 

All human societies organize, recognize and classify types of social relationships based on relations between parents, children and other descendants (consanguinity), and relations through marriage (affinity). There is also a third type applied to godparents or adoptive children (fictive). These culturally defined relationships are referred to as kinship. In many societies, it is one of the most important social organizing principles and plays a role in transmitting status and inheritance. All societies have rules of incest taboo, according to which marriage between certain kinds of kin relations is prohibited, and some also have rules of preferential marriage with certain kin relations. Pair bonding is a ubiquitous feature of human sexual relationships, whether it is manifested as serial monogamy, polygyny, or polyandry. Genetic evidence indicates that humans were predominantly polygynous for most of their existence as a species, but that this began to shift during the Neolithic, when monogamy started becoming widespread concomitantly with the transition from nomadic to sedentary societies. Anatomical evidence in the form of second-to-fourth digit ratios, a biomarker for prenatal androgen effects, likewise indicates modern humans were polygynous during the Pleistocene. 

Ethnicity 

Human ethnic groups are a social category that identifies together as a group based on shared attributes that distinguish them from other groups. These can be a common set of traditions, ancestry, language, history, society, culture, nation, religion, or social treatment within their residing area. Ethnicity is separate from the concept of race, which is based on physical characteristics, although both are socially constructed. Assigning ethnicity to a certain population is complicated, as even within common ethnic designations there can be a diverse range of subgroups, and the makeup of these ethnic groups can change over time at both the collective and individual level. Also, there is no generally accepted definition of what constitutes an ethnic group. Ethnic groupings can play a powerful role in the social identity and solidarity of ethnopolitical units. This has been closely tied to the rise of the nation state as the predominant form of political organization in the 19th and 20th centuries. 

Government and politics 

As farming populations gathered in larger and denser communities, interactions between these different groups increased. This led to the development of governance within and between the communities. Humans have evolved the ability to change affiliation with various social groups relatively easily, including previously strong political alliances, if doing so is seen as providing personal advantages. This cognitive flexibility allows individual humans to change their political ideologies, with those with higher flexibility less likely to support authoritarian and nationalistic stances. Governments create laws and policies that affect the citizens that they govern. There have been many forms of government throughout human history, each having various means of obtaining power and the ability to exert diverse controls on the population. Approximately 47% of humans live in some form of a democracy, 17% in a hybrid regime, and 37% in an authoritarian regime. Many countries belong to international organizations and alliances; the largest of these is the United Nations, with 193 member states. 

Trade and economics 

Trade, the voluntary exchange of goods and services, is seen as a characteristic that differentiates humans from other animals and has been cited as a practice that gave Homo sapiens a major advantage over other hominids. Evidence suggests early H. sapiens made use of long-distance trade routes to exchange goods and ideas, leading to cultural explosions and providing additional food sources when hunting was sparse, while such trade networks did not exist for the now extinct Neanderthals. Early trade likely involved materials for creating tools like obsidian. The first truly international trade routes were around the spice trade through the Roman and medieval periods. Early human economies were more likely to be based around gift giving instead of a bartering system. Early money consisted of commodities; the oldest being in the form of cattle and the most widely used being cowrie shells. Money has since evolved into governmental issued coins, paper and electronic money. Human study of economics is a social science that looks at how societies distribute scarce resources among different people. There are massive inequalities in the division of wealth among humans; the eight richest humans are worth the same monetary value as the poorest half of all the human population. 

Conflict 

Humans commit violence on other humans at a rate comparable to other primates, but have an increased preference for killing adults, infanticide being more common among other primates. Phylogenetic analysis predicts that 2% of early H. sapiens would be murdered, rising to 12% during the medieval period, before dropping to below 2% in modern times. There is great variation in violence between human populations, with rates of homicide about 0.01% in societies that have legal systems and strong cultural attitudes against violence. The willingness of humans to kill other members of their species en masse through organized conflict (i.e., war) has long been the subject of debate. One school of thought holds that war evolved as a means to eliminate competitors, and has always been an innate human characteristic. Another suggests that war is a relatively recent phenomenon and has appeared due to changing social conditions. While not settled, current evidence indicates that interpersonal violence has been a ubiquitous part of human history since the beginnings of the species' existence, with the bioarchaeological record showing significant incidence of violent trauma believed to be a result of war in hunter-gatherer populations long before the emergence of agriculture and settled societies. War has had a high cost on human life; it is estimated that during the 20th century, between 167 million and 188 million people died as a result of war. War casualty data is less reliable for pre-medieval times, especially global figures. But compared with any period over the past 600 years, the past 80 years (post-1945) have seen a very significant drop in global military and civilian death rates due to armed conflict. 

See also 

List of human evolution fossils Timeline of human evolution 

Notes 

References 

[Life] Life is the capacity in matter, formed of one or more units called cells, for processes such as cell signaling, homeostasis, metabolism, cell growth, adaptation, response to stimuli, and reproduction. All life eventually reaches a state of death. Many philosophical definitions of living systems have been proposed, such as self-organizing systems. Defining life is further complicated by viruses, which replicate only in host cells, and the possibility of extraterrestrial life, which could be very different from life on Earth. Life exists all over the Earth in air, water, and soil, with many ecosystems forming the biosphere. Some of these are harsh environments occupied only by extremophiles. The life in a particular ecosystem is called its biota. Life has been studied since ancient times, with theories such as Empedocles's materialism asserting that it was composed of four eternal elements, and Aristotle's hylomorphism asserting that living things have souls and embody both form and matter. Life originated at least 3.5 billion years ago, resulting in a universal common ancestor. This evolved into all the species that exist now, by way of many extinct species, some of which have left traces as fossils. Attempts to classify living things, too, began with Aristotle. Modern classification began with Carl Linnaeus's system of binomial nomenclature in the 1740s. Living things are composed of biochemical molecules, formed mainly from a few core chemical elements. All living things contain two types of macromolecule, proteins and nucleic acids, the latter usually both DNA and RNA: these carry the information needed by each species, including the instructions to make each type of protein. The proteins, in turn, serve as the machinery which carries out the many chemical processes of life. The cell is the structural and functional unit of life. Smaller organisms, including prokaryotes (bacteria and archaea), consist of small single cells. Larger organisms, mainly eukaryotes, can consist of single cells or may be multicellular with more complex structure. Life is only known to exist on Earth but extraterrestrial life is thought probable. Artificial life is being simulated and explored by scientists and engineers. 

Definitions 

Challenge The definition of life has long been a challenge for scientists and philosophers. This is partially because life is a process, not a substance. This is complicated by a lack of knowledge of the characteristics of living entities, if any, that may have developed outside Earth. Philosophical definitions of life have also been put forward, with similar difficulties on how to distinguish living things from the non-living. Legal definitions of life have been debated, though these generally focus on the decision to declare a human dead, and the legal ramifications of this decision. At least 123 definitions of life have been compiled. A biota is the assemblage of living things, especially the animals and plants, that inhabit a specific place and time, such as an ecosystem or biome; thus, the goal of nature conservation is to preserve the biota of an ecosystem. 

Descriptive 

Since there is no consensus for a definition of life, most current definitions in biology, the scientific study of life, are descriptive. Life is considered a characteristic of something that preserves, furthers or reinforces its existence in the given environment. This implies all or most of the following traits: 

Homeostasis: regulation of the internal environment to maintain a constant state; for example, sweating to reduce temperature. Organisation: being structurally composed of one or more cells – the basic units of life. Metabolism: transformation of energy, used to convert chemicals into cellular components (anabolism) and to decompose organic matter (catabolism). Living things require energy for homeostasis and other activities. Growth: maintenance of a higher rate of anabolism than catabolism. A growing organism increases in size and structure. Adaptation: the evolutionary process whereby an organism becomes better able to live in its habitat. Response to stimuli: such as the contraction of a unicellular organism away from external chemicals, the complex reactions involving all the senses of multicellular organisms, or the motion of the leaves of a plant turning toward the sun (phototropism), and chemotaxis. Reproduction: the ability to produce new individual organisms, either asexually from a single parent organism or sexually from two parent organisms. 

Physics 

From a physics perspective, an organism is a thermodynamic system with an organised molecular structure that can reproduce itself and evolve as survival dictates. Thermodynamically, life has been described as an open system which makes use of gradients in its surroundings to create imperfect copies of itself. Another way of putting this is to define life as "a self-sustained chemical system capable of undergoing Darwinian evolution", a definition adopted by a NASA committee attempting to define life for the purposes of exobiology, based on a suggestion by Carl Sagan. This definition, however, has been widely criticised because according to it, a single sexually reproducing individual is not alive as it is incapable of evolving on its own. 

Living systems 

Others take a living systems theory viewpoint that does not necessarily depend on molecular chemistry. One systemic definition of life is that living things are self-organizing and autopoietic (self-producing). Variations of this include Stuart Kauffman's definition as an autonomous agent or a multi-agent system capable of reproducing itself, and of completing at least one thermodynamic work cycle. This definition is extended by the evolution of novel functions over time. Living systems are characterized by a multiscale, hierarchical organization, spanning from molecular machines to cells, organs, tissues, organisms, populations, ecosystems, up to the whole biosphere. 

Death 

Death is the termination of all vital functions or life processes in an organism or cell. One of the challenges in defining death is in distinguishing it from life. Death would seem to refer to either the moment life ends, or when the state that follows life begins. However, determining when death has occurred is difficult, as cessation of life functions is often not simultaneous across organ systems. Such determination, therefore, requires drawing conceptual lines between life and death. This is problematic because there is little consensus over how to define life. The nature of death has for millennia been a central concern of the world's religious traditions and of philosophical inquiry. Many religions maintain faith in either a kind of afterlife or reincarnation for the soul, or resurrection of the body at a later date. 

Viruses 

Whether or not viruses should be considered as alive is controversial. They are most often considered as just gene coding replicators rather than forms of life. They have been described as "organisms at the edge of life" because they possess genes, evolve by natural selection, and replicate by making multiple copies of themselves through self-assembly. However, viruses do not metabolise and they require a host cell to make new products. Virus self-assembly within host cells has implications for the study of the origin of life, as it may support the hypothesis that life could have started as self-assembling organic molecules. 

History of study 

Materialism 

Some of the earliest theories of life were materialist, holding that all that exists is matter, and that life is merely a complex form or arrangement of matter. Empedocles (430 BC) argued that everything in the universe is made up of a combination of four eternal "elements" or "roots of all": earth, water, air, and fire. All change is explained by the arrangement and rearrangement of these four elements. The various forms of life are caused by an appropriate mixture of elements. Democritus (460 BC) was an atomist; he thought that the essential characteristic of life was having a soul (psyche), and that the soul, like everything else, was composed of fiery atoms. He elaborated on fire because of the apparent connection between life and heat, and because fire moves. Plato, in contrast, held that the world was organised by permanent forms, reflected imperfectly in matter; forms provided direction or intelligence, explaining the regularities observed in the world. The mechanistic materialism that originated in ancient Greece was revived and revised by the French philosopher René Descartes (1596–1650), who held that animals and humans were assemblages of parts that together functioned as a machine. Gottfried Wilhelm Leibniz emphasised the hierarchical organization of living machines, noting in his book Monadology (1714) that "...the machines of nature, that is living bodies, are still machines in their smallest parts, to infinity." This idea was developed further by Julien Offray de La Mettrie (1709–1750) in his book L'Homme Machine. In the 19th century the advances in cell theory in biological science encouraged this view. The evolutionary theory of Charles Darwin (1859) is a mechanistic explanation for the origin of species by means of natural selection. At the beginning of the 20th century Stéphane Leduc (1853–1939) promoted the idea that biological processes could be understood in terms of physics and chemistry, and that their growth resembled that of inorganic crystals immersed in solutions of sodium silicate. His ideas, set out in his book La biologie synthétique, were widely dismissed during his lifetime, but has incurred a resurgence of interest in the work of Russell, Barge and colleagues. 

Hylomorphism 

Hylomorphism is a theory first expressed by the Greek philosopher Aristotle (322 BC). The application of hylomorphism to biology was important to Aristotle, and biology is extensively covered in his extant writings. In this view, everything in the material universe has both matter and form, and the form of a living thing is its soul (Greek psyche, Latin anima). There are three kinds of souls: the vegetative soul of plants, which causes them to grow and decay and nourish themselves, but does not cause motion and sensation; the animal soul, which causes animals to move and feel; and the rational soul, which is the source of consciousness and reasoning, which (Aristotle believed) is found only in man. Each higher soul has all of the attributes of the lower ones. Aristotle believed that while matter can exist without form, form cannot exist without matter, and that therefore the soul cannot exist without the body. This account is consistent with teleological explanations of life, which account for phenomena in terms of purpose or goal-directedness. Thus, the whiteness of the polar bear's coat is explained by its purpose of camouflage. The direction of causality (from the future to the past) is in contradiction with the scientific evidence for natural selection, which explains the consequence in terms of a prior cause. Biological features are explained not by looking at future optimal results, but by looking at the past evolutionary history of a species, which led to the natural selection of the features in question. 

Spontaneous generation 

Spontaneous generation was the belief that living organisms can form without descent from similar organisms. Typically, the idea was that certain forms such as fleas could arise from inanimate matter such as dust or the supposed seasonal generation of mice and insects from mud or garbage. The theory of spontaneous generation was proposed by Aristotle, who compiled and expanded the work of prior natural philosophers and the various ancient explanations of the appearance of organisms; it was considered the best explanation for two millennia. It was decisively dispelled by the experiments of Louis Pasteur in 1859, who expanded upon the investigations of predecessors such as Francesco Redi. Disproof of the traditional ideas of spontaneous generation is no longer controversial among biologists. 

Vitalism 

Vitalism is the belief that there is a non-material life-principle. This originated with Georg Ernst Stahl (17th century), and remained popular until the middle of the 19th century. It appealed to philosophers such as Henri Bergson, Friedrich Nietzsche, and Wilhelm Dilthey, anatomists like Xavier Bichat, and chemists like Justus von Liebig. Vitalism included the idea that there was a fundamental difference between organic and inorganic material, and the belief that organic material can only be derived from living things. This was disproved in 1828, when Friedrich Wöhler prepared urea from inorganic materials. This Wöhler synthesis is considered the starting point of modern organic chemistry. It is of historical significance because for the first time an organic compound was produced in inorganic reactions. During the 1850s Hermann von Helmholtz, anticipated by Julius Robert von Mayer, demonstrated that no energy is lost in muscle movement, suggesting that there were no "vital forces" necessary to move a muscle. These results led to the abandonment of scientific interest in vitalistic theories, especially after Eduard Buchner's demonstration that alcoholic fermentation could occur in cell-free extracts of yeast. Nonetheless, belief still exists in pseudoscientific theories such as homoeopathy, which interprets diseases and sickness as caused by disturbances in a hypothetical vital force or life force. 

Development 

Origin of life 

The age of Earth is about 4.54 billion years. Life on Earth has existed for at least 3.5 billion years, with the oldest physical traces of life dating back 3.7 billion years. Estimates from molecular clocks, as summarised in the TimeTree public database, place the origin of life around 4 billion years ago. Hypotheses on the origin of life attempt to explain the formation of a universal last common ancestor from simple organic molecules via pre-cellular life to protocells and metabolism. In 2016, a set of 355 genes from the last universal common ancestor was tentatively identified. The biosphere is postulated to have developed, from the origin of life onwards, at least some 3.5 billion years ago. The earliest evidence for life on Earth includes biogenic graphite found in 3.7 billion-year-old metasedimentary rocks from Western Greenland and microbial mat fossils found in 3.48 billion-year-old sandstone from Western Australia. More recently, in 2015, "remains of biotic life" were found in 4.1 billion-year-old rocks in Western Australia. In 2017, putative fossilised microorganisms (or microfossils) were announced to have been discovered in hydrothermal vent precipitates in the Nuvvuagittuq Belt of Quebec, Canada that were as old as 4.28 billion years, the oldest record of life on Earth, suggesting "an almost instantaneous emergence of life" after ocean formation 4.4 billion years ago, and not long after the formation of the Earth 4.54 billion years ago. 

Evolution 

Evolution is the change in heritable characteristics of biological populations over successive generations. It results in the appearance of new species and often the disappearance of old ones. Evolution occurs when evolutionary processes such as natural selection (including sexual selection) and genetic drift act on genetic variation, resulting in certain characteristics increasing or decreasing in frequency within a population over successive generations. The process of evolution has given rise to biodiversity at every level of biological organisation. 

Fossils 

Fossils are the preserved remains or traces of organisms from the remote past. The totality of fossils, both discovered and undiscovered, and their placement in layers (strata) of sedimentary rock is known as the fossil record. A preserved specimen is called a fossil if it is older than the arbitrary date of 10,000 years ago. Hence, fossils range in age from the youngest at the start of the Holocene Epoch to the oldest from the Archaean Eon, up to 3.4 billion years old. 

Extinction 

Extinction is the process by which a species dies out. The moment of extinction is the death of the last individual of that species. Because a species' potential range may be very large, determining this moment is difficult, and is usually done retrospectively after a period of apparent absence. Species become extinct when they are no longer able to survive in changing habitat or against superior competition. Over 99% of all the species that have ever lived are now extinct. Mass extinctions may have accelerated evolution by providing opportunities for new groups of organisms to diversify. 

Environmental conditions 

The diversity of life on Earth is a result of the dynamic interplay between genetic opportunity, metabolic capability, environmental challenges, and symbiosis. For most of its existence, Earth's habitable environment has been dominated by microorganisms and subjected to their metabolism and evolution. As a consequence of these microbial activities, the physical-chemical environment on Earth has been changing on a geologic time scale, thereby affecting the path of evolution of subsequent life. For example, the release of molecular oxygen by cyanobacteria as a by-product of photosynthesis induced global changes in the Earth's environment. Because oxygen was toxic to most life on Earth at the time, this posed novel evolutionary challenges, and ultimately resulted in the formation of Earth's major animal and plant species. This interplay between organisms and their environment is an inherent feature of living systems. 

Biosphere 

The biosphere is the global sum of all ecosystems. It can also be termed as the zone of life on Earth, a closed system (apart from solar and cosmic radiation and heat from the interior of the Earth), and largely self-regulating. Organisms exist in every part of the biosphere, including soil, hot springs, inside rocks at least 19 km (12 mi) deep underground, the deepest parts of the ocean, and at least 64 km (40 mi) high in the atmosphere. For example, spores of Penicillium notatum have been detected in the mesosphere at an altitude of 57 to 77 km. Under test conditions, life forms have been observed to survive in the vacuum of space. Life forms thrive in the deep Mariana Trench, and inside rocks up to 580 m (1,900 ft; 0.36 mi) below the sea floor under 2,590 m (8,500 ft; 1.61 mi) of ocean off the coast of the northwestern United States, and 2,400 m (7,900 ft; 1.5 mi) beneath the seabed off Japan. There are even bacteria that thrive in nuclear reactors. In 2014, life forms were found living 800 m (2,600 ft; 0.50 mi) below the ice of Antarctica. Expeditions of the International Ocean Discovery Program found unicellular life in 120 °C sediment 1.2 km below seafloor in the Nankai Trough subduction zone. According to one researcher, "You can find microbes everywhere—they're extremely adaptable to conditions, and survive wherever they are." 

Range of tolerance The inert components of an ecosystem are the physical and chemical factors necessary for life—energy (sunlight or chemical energy), water, heat, atmosphere, gravity, nutrients, and ultraviolet solar radiation protection. In most ecosystems, the conditions vary during the day and from one season to the next. To survive in these ecosystems, organisms must be able to tolerate a range of conditions defined as the "range of tolerance". Outside this range are the "zones of physiological stress", where the survival and reproduction are possible but not optimal. Beyond these zones are the "zones of intolerance", where survival and reproduction of that organism is unlikely or impossible. Organisms that have a wide range of tolerance are more widely distributed than organisms with a narrow range of tolerance. 

Extremophiles 

To survive, some microorganisms have evolved to withstand freezing, complete desiccation, starvation, high levels of radiation exposure, and other physical or chemical challenges. These extremophile microorganisms may survive exposure to such conditions for long periods. They excel at exploiting uncommon sources of energy. Characterization of the structure and metabolic diversity of microbial communities in such extreme environments is ongoing. 

Classification 

Western Antiquity 

The first classification of organisms was made by the Greek philosopher Aristotle (384–322 BC), who grouped living things as either plants or animals, based mainly on their ability to move. He distinguished animals with blood from animals without blood, which can be compared with the concepts of vertebrates and invertebrates respectively, and divided the blooded animals into five groups: viviparous quadrupeds (mammals), oviparous quadrupeds (reptiles and amphibians), birds, fishes and whales. The bloodless animals were divided into five groups: cephalopods, crustaceans, insects (which included the spiders, scorpions, and centipedes), shelled animals (such as most molluscs and echinoderms), and zoophytes (animals that resembled plants). This theory remained dominant for more than a thousand years. 

Eastern philosophy In ancient Indian medicine, the Sushruta Samhita (traditionally dated to around the 6th century BCE) classified poisons, and by extension substances more broadly, into sthavara ("immobile"), of plant or mineral origin, and jangama ("mobile" or animate), of animal origin. The plant kingdom itself was further subdivided by the text into categories such as vriksha (fruit- and flower-bearing trees) and virudha (creepers and shrubs). The companion Charaka Samhita used a comparable three-way division of sthavara, jangama, and artificial (samyogaja) categories. In ancient China, the Erya, the oldest surviving Chinese dictionary, possibly assembled during the Qin or early Han dynasty from material dating to the Zhou dynasty, organizes its entries into 19 semantic categories, including separate sections on grasses, trees, insects, fish, birds, and beasts. Later Chinese materia medica works, culminating in Li Shizhen's Bencao Gangmu (1578), organized substances by medicinal use into plant-associated and animal-associated categories. 

Linnaean In the late 1740s, Carl Linnaeus introduced his system of binomial nomenclature for the classification of species. Linnaeus attempted to improve the composition and reduce the length of the previously used many-worded names by abolishing unnecessary rhetoric, introducing new descriptive terms and precisely defining their meaning. The fungi were originally treated as plants. For a short period Linnaeus had classified them in the taxon Vermes in Animalia, but later placed them back in Plantae. Herbert Copeland classified the Fungi in his Protoctista, including them with single-celled organisms and thus partially avoiding the problem but acknowledging their special status. The problem was eventually solved by Whittaker, when he gave them their own kingdom in his five-kingdom system. Evolutionary history shows that the fungi are more closely related to animals than to plants. As advances in microscopy enabled detailed study of cells and microorganisms, new groups of life were revealed, and the fields of cell biology and microbiology were created. These new organisms were originally described separately in protozoa as animals and protophyta/thallophyta as plants, but were united by Ernst Haeckel in the kingdom Protista; later, the prokaryotes were split off in the kingdom Monera, which would eventually be divided into two separate groups, the Bacteria and the Archaea. This led to the six-kingdom system and eventually to the current three-domain system, which is based on evolutionary relationships. However, the classification of eukaryotes, especially of protists, is still controversial. As microbiology developed, viruses, which are non-cellular, were discovered. Whether these are considered alive has been a matter of debate; viruses lack characteristics of life such as cell membranes, metabolism and the ability to grow or respond to their environments. Viruses have been classed into "species" based on their genetics, but many aspects of such a classification remain controversial. The original Linnaean system has been modified many times, for example as follows: 

The attempt to organise the Eukaryotes into a small number of kingdoms has been challenged. The Protozoa do not form a clade or natural grouping, and nor do the Chromista (Chromalveolata). 

Metagenomic The ability to sequence large numbers of complete genomes has allowed biologists to take a metagenomic view of the phylogeny of the whole tree of life. This has led to the realisation that the majority of living things are bacteria, and that all have a common origin. 

Composition 

Chemical elements All life forms require certain core chemical elements for their biochemical functioning. These include carbon, hydrogen, nitrogen, oxygen, phosphorus, and sulfur—the elemental macronutrients for all organisms. Together these make up nucleic acids, proteins and lipids, the bulk of living matter. Five of these six elements comprise the chemical components of DNA, the exception being sulfur. The latter is a component of the amino acids cysteine and methionine. The most abundant of these elements in organisms is carbon, which has the desirable attribute of forming multiple, stable covalent bonds. This allows carbon-based (organic) molecules to form the immense variety of chemical arrangements described in organic chemistry. Alternative hypothetical types of biochemistry have been proposed that eliminate one or more of these elements, swap out an element for one not on the list, or change required chiralities or other chemical properties. 

DNA 

Deoxyribonucleic acid or DNA is a molecule that carries most of the genetic instructions used in the growth, development, functioning and reproduction of all known living organisms and many viruses. DNA and RNA are nucleic acids; alongside proteins and complex carbohydrates, they are one of the three major types of macromolecule that are essential for all known forms of life. Most DNA molecules consist of two biopolymer strands coiled around each other to form a double helix. The two DNA strands are known as polynucleotides since they are composed of simpler units called nucleotides. Each nucleotide is composed of a nitrogen-containing nucleobase—either cytosine (C), guanine (G), adenine (A), or thymine (T)—as well as a sugar called deoxyribose and a phosphate group. The nucleotides are joined to one another in a chain by covalent bonds between the sugar of one nucleotide and the phosphate of the next, resulting in an alternating sugar-phosphate backbone. According to base pairing rules (A with T, and C with G), hydrogen bonds bind the nitrogenous bases of the two separate polynucleotide strands to make double-stranded DNA. This has the key property that each strand contains all the information needed to recreate the other strand, enabling the information to be preserved during reproduction and cell division. Within cells, DNA is organised into long structures called chromosomes. During cell division these chromosomes are duplicated in the process of DNA replication, providing each cell its own complete set of chromosomes. Eukaryotes store most of their DNA inside the cell nucleus.  

Cells 

Cells are the basic unit of structure in every living thing, and all cells arise from pre-existing cells by division. Cell theory was formulated by Henri Dutrochet, Theodor Schwann, Rudolf Virchow and others during the early nineteenth century, and subsequently became widely accepted. The activity of an organism depends on the total activity of its cells, with energy flow occurring within and between them. Cells contain hereditary information that is carried forward as a genetic code during cell division. There are two primary types of cells, reflecting their evolutionary origins. Prokaryote cells lack a nucleus and other membrane-bound organelles, although they have circular DNA and ribosomes. Bacteria and Archaea are two domains of prokaryotes. The other primary type is the eukaryote cell, which has a distinct nucleus bound by a nuclear membrane and membrane-bound organelles, including mitochondria, chloroplasts, lysosomes, rough and smooth endoplasmic reticulum, and vacuoles. In addition, their DNA is organised into chromosomes. All species of large complex organisms are eukaryotes, including animals, plants and fungi, though with a wide diversity of protist microorganisms. The conventional model is that eukaryotes evolved from prokaryotes, with the main organelles of the eukaryotes forming through endosymbiosis between bacteria and the progenitor eukaryotic cell. The molecular mechanisms of cell biology are based on proteins. Most of these are synthesised by the ribosomes through an enzyme-catalysed process called protein biosynthesis. A sequence of amino acids is assembled and joined based upon gene expression of the cell's nucleic acid. In eukaryotic cells, these proteins may then be transported and processed through the Golgi apparatus in preparation for dispatch to their destination. Cells reproduce through a process of cell division in which the parent cell divides into two or more daughter cells. For prokaryotes, cell division occurs through a process of fission in which the DNA is replicated, then the two copies are attached to parts of the cell membrane. In eukaryotes, a more complex process of mitosis is followed. However, the result is the same; the resulting cell copies are identical to each other and to the original cell (except for mutations), and both are capable of further division following an interphase period. Most species of multicellular plants, animals and fungi as well as many protists are capable of sexual reproduction. Sexual reproduction, involving a meiotic process, is considered to have arisen very early in the evolution of eukaryotes. 

Multicellular structure Multicellular organisms may have first evolved through the formation of colonies of identical cells. These cells can form group organisms through cell adhesion. The individual members of a colony are capable of surviving on their own, whereas the members of a true multi-cellular organism have developed specialisations, making them dependent on the remainder of the organism for survival. Such organisms are formed clonally or from a single germ cell that is capable of forming the various specialised cells that form the adult organism. This specialisation allows multicellular organisms to exploit resources more efficiently than single cells. About 800 million years ago, a minor genetic change in a single molecule, the enzyme GK-PID, may have allowed organisms to go from a single cell organism to one of many cells. Cells have evolved methods to perceive and respond to their microenvironment, thereby enhancing their adaptability. Cell signalling coordinates cellular activities, and hence governs the basic functions of multicellular organisms. Signaling between cells can occur through direct cell contact using juxtacrine signalling, or indirectly through the exchange of agents as in the endocrine system. In more complex organisms, coordination of activities can occur through a dedicated nervous system. 

In the universe 

Though life is confirmed only on Earth, many think that extraterrestrial life is not only plausible, but probable or inevitable, possibly resulting in a biophysical cosmology instead of a mere physical cosmology. Other planets and moons in the Solar System and other planetary systems are being examined for evidence of having once supported simple life, and projects such as SETI are trying to detect radio transmissions from possible alien civilisations. Other locations within the Solar System that may host microbial life include the subsurface of Mars, the upper atmosphere of Venus, and subsurface oceans on some of the moons of the giant planets. Investigation of the tenacity and versatility of life on Earth, as well as an understanding of the molecular systems that some organisms utilise to survive such extremes, is important for the search for extraterrestrial life. For example, lichen could survive for a month in a simulated Martian environment. Beyond the Solar System, the region around another main-sequence star that could support Earth-like life on an Earth-like planet is known as the habitable zone. The inner and outer radii of this zone vary with the luminosity of the star, as does the time interval during which the zone survives. Stars more massive than the Sun have a larger habitable zone, but remain on the Sun-like "main sequence" of stellar evolution for a shorter time interval. Small red dwarfs have the opposite problem, with a smaller habitable zone that is subject to higher levels of magnetic activity and the effects of tidal locking from close orbits. Hence, stars in the intermediate mass range such as the Sun may have a greater likelihood for Earth-like life to develop. The location of the star within a galaxy may also affect the likelihood of life forming. Stars in regions with a greater abundance of heavier elements that can form planets, in combination with a low rate of potentially habitat-damaging supernova events, are predicted to have a higher probability of hosting planets with complex life. The variables of the Drake equation are used to discuss the conditions in planetary systems where civilisation is most likely to exist, within wide bounds of uncertainty. A "Confidence of Life Detection" scale (CoLD) for reporting evidence of life beyond Earth has been proposed. 

Artificial 

Artificial life is the simulation of any aspect of life, as through computers, robotics, or biochemistry. Synthetic biology is a new area of biotechnology that combines science and biological engineering. The common goal is the design and construction of new biological functions and systems not found in nature. Synthetic biology includes the broad redefinition and expansion of biotechnology, with the ultimate goals of being able to design and build engineered biological systems that process information, manipulate chemicals, fabricate materials and structures, produce energy, provide food, and maintain and enhance human health and the environment. 

See also 

Notes 

References 

External links 

Vitae (BioLib) Wikispecies – a free directory of life Biota (Taxonomicon) (archived 15 July 2014) Entry on the Stanford Encyclopedia of Philosophy What Is Life? – by Jaime Green, The Atlantic (archived 5 December 2023) 

[Universe] The universe comprises all of existence: all forms of matter and energy, and the structures they form, from subatomic particles to entire galactic filaments. Since the early 20th century, the field of cosmology has established that the universe has been expanding for 13.8 billion years, starting from a dense fireball in an event called the Big Bang. The observable portion of the universe is approximately 93 billion light-years in diameter at present. The total size of the universe is not known. Some of the earliest cosmological models of the universe were geocentric, placing Earth at the center. During the Scientific Revolution, astronomical observations led to a heliocentric model. Further observational improvements led to the realization that the Sun is one of a few hundred billion stars in the Milky Way, which is one of a few hundred billion galaxies in the observable universe. At the largest scale, galaxies are distributed uniformly and the same in all directions. At smaller scales, galaxies are distributed in clusters and superclusters, which form immense filaments and voids in space, creating a vast foam-like structure. Discoveries in the early 20th century, including general relativity, led to the modern view of an expanding, isotropic, homogeneous universe. Evidence accumulated supporting the Big Bang theory: an initial hot fireball cooled and becoming less dense as the universe expanded, allowing the first subatomic particles and simple atoms to form. Giant clouds of hydrogen and helium were gradually drawn to the places where matter was most dense, forming the first galaxies, stars, and eventually, everything else. From studying the effects of gravity on both matter and light, it has been discovered that the universe contains much more matter than is accounted for by visible objects; stars, galaxies, nebulae and interstellar gas. This unseen matter is known as dark matter. In the widely accepted ΛCDM cosmological model, dark matter accounts for about 25.8%±1.1% of the mass and energy in the universe while about 69.2%±1.2% is dark energy, a mysterious form of energy responsible for the acceleration of the expansion of the universe. Ordinary ('baryonic') matter therefore composes only 4.84%±0.1% of the universe. Stars, planets, and visible gas clouds only form about 6% of this ordinary matter. There are many competing hypotheses about the ultimate fate of the universe and about what, if anything, preceded the Big Bang. 

Definition 

The physical universe has been defined as "The totality of all space and time; all that is, has been, and will be." The universe contains all energy and matter, including therefore planets, moons, stars, galaxies, and the contents of intergalactic space. Some philosophers and scientists support the inclusion of ideas and abstract concepts—such as mathematics and logic—in the definition of the universe. The word universe may also refer to concepts such as the cosmos, the world, and nature. 

Etymology The word universe derives from the Old French word univers, which in turn derives from the Latin word universus, meaning 'combined into one'. The Latin word 'universum' was used by Cicero and later Latin authors in many of the same senses as the modern English word is used. 

Synonyms A term for universe among the ancient Greek philosophers from Pythagoras onwards was τὸ πᾶν (tò pân) 'the all', defined as all matter and all space, and τὸ ὅλον (tò hólon) 'all things', which did not necessarily include the void. Another synonym was ὁ κόσμος (ho kósmos) meaning 'the world, the cosmos'. Synonyms are also found in Latin authors (totum, mundus, natura) and survive in modern languages, e.g., the German words Das All, Weltall, and Natur for universe. The same synonyms are found in English, such as everything (as in the theory of everything), the cosmos (as in cosmology), the world (as in the many-worlds interpretation), and nature (as in natural laws or natural philosophy). 

Chronology and the Big Bang 

The prevailing model for the evolution of the universe is the Big Bang theory. In the Big Bang model, the earliest state of the universe was extremely hot and dense but the universe cooled during subsequent expansion. The model is based on general relativity and on symmetry assumptions such as the homogeneity and isotropy of space. A version of the model with a cosmological constant (Lambda) and cold dark matter, known as the Lambda-CDM model, provides an excellent account of most observations of the universe. 

Much of very earliest time is not understood. An intense period of expansion called cosmic inflation is postulated to explain many astronomical observations and set the initial conditions for the Lambda-CDM model. Within the first fraction of a second of the universe's existence, it was extremely dense, and the high energy meant all the particles of the Standard model were in equilibrium. As the universe cooled due to expansion, the state of the universe went through phase transitions analogous to water freezing. Various types of elementary particles associated stably producing a plasma of electrons, protons, and neutrons, with very energetic photons preventing them from binding until about one minute after the Big Bang. During the next few minutes, some protons and neutrons combined to form atomic nuclei through nuclear fusion. This process, known as Big Bang nucleosynthesis, lasted for about 15 minutes, produced helium, with small amounts of deuterium (a form of hydrogen) and traces of lithium. No other nuclei formed in significant amounts during this time. All of the neutrons that did not fuse decayed in to protons and electrons. After nucleosynthesis ended, the universe was still far too hot for matter to form neutral atoms, so it contained a hot, dense, optically opaque plasma of negatively charged electrons, neutral neutrinos and positive nuclei. After about 377,000 years, the universe had cooled enough that electrons and nuclei could form the first stable atoms. This is known as recombination for historical reasons; electrons and nuclei were combining for the first time. Unlike plasma, neutral atoms are transparent to many wavelengths of light, so for the first time, the universe also became transparent. The photons released ("decoupled") when these atoms formed can still be seen today; they form the cosmic microwave background (CMB). As the universe expanded, the energy density of electromagnetic radiation decreased more quickly than that of matter because the energy of each photon decreased as it is cosmologically redshifted. At around 47,000 years, the energy density of matter became larger than that of photons and neutrinos, and began to dominate the large scale behavior of the universe. This marked the end of the radiation-dominated era and the start of the matter-dominated era. In the earliest stages of the universe, tiny fluctuations within the universe's density led to concentrations of dark matter gradually forming. Ordinary matter, attracted to these by gravity, formed large gas clouds and eventually, stars and galaxies, where the dark matter was most dense, and voids where it was least dense. After around 100–300 million years, the first stars formed, known as Population III stars. These were probably very massive, luminous, non metallic and short-lived. They were responsible for the gradual reionization of the universe between about 200–500 million years and 1 billion years, and also for seeding the universe with elements heavier than helium, through stellar nucleosynthesis. The universe contains a mysterious energy—possibly a scalar field—called dark energy, the density of which does not change over time. After about 9.8 billion years, the universe had expanded sufficiently so that the density of matter was less than the density of dark energy, marking the beginning of the present dark-energy-dominated era. In this era, the expansion of the universe is accelerating due to dark energy. 

Physical properties The properties of the universe are measured by observational cosmology, a combination of direct observations and observations analyzed in conjunction with models based on general relativity and Einstein's field equations. 

Size 

Due to the finite speed of light, there is a limit (known as the particle horizon) to how far light can travel over the age of the universe. The spatial region from which we can receive light is called the observable universe. The proper distance (measured at a fixed time) between Earth and the edge of the observable universe is 46 billion light-years (14 billion parsecs), making the diameter of the observable universe about 93 billion light-years (28 billion parsecs). Although the distance traveled by light from the edge of the observable universe is close to the age of the universe times the speed of light, 13.8 billion light-years (4.2×10^9 pc), the proper distance is larger because the edge of the observable universe and the Earth have since moved further apart. For comparison, the Milky Way is roughly 87,400 light-years in diameter, and the nearest sister galaxy to the Milky Way, the Andromeda Galaxy, is located roughly 2.5 million light-years away. Because humans cannot observe space beyond the edge of the observable universe, it is unknown whether the size of the universe in its totality is finite or infinite. 

Age and expansion 

Assuming that the Lambda-CDM model is correct, the measurements of the parameters using a variety of techniques by numerous experiments yield a best value of the age of the universe at 13.799 ± 0.021 billion years, as of 2015. Over time, the universe and its contents have evolved. For example, the relative population of quasars and galaxies has changed and the universe has expanded. This expansion is inferred from the observation that the light from distant galaxies has been redshifted, which implies that the galaxies are receding from us. Analyses of Type Ia supernovae indicate that the expansion is accelerating. The more matter there is in the universe, the stronger the mutual gravitational pull of the matter. If the universe were too dense then it would re-collapse into a black hole. However, if the universe contained too little matter then it would expand too quickly for astronomical structures, like galaxies or planets, to form. Since the Big Bang, the universe has expanded monotonically. The mass–energy density of the universe, equivalent to about 5 protons per cubic meter, allowed it to expand for the last 13.8 billion years, giving time to form the universe as observed today. There are dynamical forces acting on the particles in the universe which affect the expansion rate. Before 1998, it was expected that the expansion rate would be decreasing as time went on due to the influence of gravitational interactions in the universe; and thus there is an additional observable quantity in the universe called the deceleration parameter, which most cosmologists expected to be positive and related to the matter density of the universe. In 1998, the deceleration parameter was measured by two different groups to be negative, approximately −0.55, which technically implies that the second derivative of the cosmic scale factor  

         a 
          ¨ 
         
       
     
   
 
{\displaystyle {\ddot {a}}} 
  

has been positive in the last 5–6 billion years. 

Spacetime 

Modern physics regards events as being organized into spacetime. This idea originated with the special theory of relativity, which predicts that if one observer sees two events happening in different places at the same time, a second observer who is moving relative to the first will see those events happening at different times. The two observers will disagree on the time  

   T 
   
 
{\displaystyle T} 
  

between the events, and they will disagree about the distance  

   D 
   
 
{\displaystyle D} 
  

separating the events, but they will agree on the speed of light  

   c 
   
 
{\displaystyle c} 
  

, and they will measure the same value for the combination  

     c 
       
        2 
       
     
     
      T 
       
        2 
       
     
    − 
     
      D 
       
        2 
       
     
   
 
{\displaystyle c^{2}T^{2}-D^{2}} 
  

. The square root of the absolute value of this quantity is called the interval between the two events. The interval expresses how widely separated events are, not just in space or in time, but in the combined setting of spacetime. The special theory of relativity describes a flat spacetime. Its successor, the general theory of relativity, explains gravity as curvature of spacetime arising due to its energy content. A curved path like an orbit is not the result of a force deflecting a body from an ideal straight-line path, but rather the body's attempt to fall freely through a background that is itself curved by the presence of other masses. A remark by John Archibald Wheeler that has become proverbial among physicists summarizes the theory: "Spacetime tells matter how to move; matter tells spacetime how to curve", and therefore there is no point in considering one without the other. The Newtonian theory of gravity is a good approximation to the predictions of general relativity when gravitational effects are weak and objects are moving slowly compared to the speed of light. The relation between matter distribution and spacetime curvature is given by the Einstein field equations, which require tensor calculus to express. The universe appears to be a smooth spacetime continuum consisting of three spatial dimensions and one temporal (time) dimension. Therefore, an event in the spacetime of the physical universe can be identified by a set of four coordinates: (x, y, z, t). 

Shape 

Cosmologists often work with space-like slices of spacetime that are surfaces of constant time in comoving coordinates. The geometry of these spatial slices is set by the density parameter, Omega (Ω), defined as the average matter density of the universe divided by a critical value. This selects one of three possible geometries depending on whether Ω is equal to, less than, or greater than 1. These are called, respectively, the flat, open and closed universes. Observations, including the Cosmic Background Explorer (COBE), Wilkinson Microwave Anisotropy Probe (WMAP), and Planck maps of the CMB, suggest that the universe is infinite in extent with a finite age, as described by the Friedmann–Lemaître–Robertson–Walker (FLRW) models. These FLRW models thus support inflationary models and the standard model of cosmology, describing a flat, homogeneous universe presently dominated by dark matter and dark energy. 

Support of life The frequency of life in the universe has been a frequent point of investigation in astronomy and astrobiology, being the issue of the Drake equation and the different views on it, from identifying the Fermi paradox, the situation of not having found any signs of extraterrestrial life, to arguments for a biophysical cosmology, a view of life being inherent to the physical cosmology of the universe. The fine-tuned universe hypothesis is the proposition that the conditions that allow the existence of observable life in the universe can only occur when certain universal fundamental physical constants lie within a very narrow range of values. According to this hypothesis, if any of several fundamental constants were only slightly different, the universe would have been unlikely to be conducive to the establishment and development of matter, astronomical structures, elemental diversity, or life as it is understood. Whether this is true, and whether that question is even logically meaningful to ask, are subjects of much debate. The proposition is discussed among philosophers, scientists, theologians, and proponents of creationism. 

Composition 

The mass–energy density of the universe is 68% dark energy, 27% dark matter, and 5% ordinary matter. Other contents are neutrinos (less than 0.3%) and electromagnetic radiation (about 0.005%). The universe has 10 billion times more matter than antimatter. In the very early universe matter and antimatter annihilated each other leaving a high density of photons. In the Standard Model of particle physics, equal amounts antimatter and matter should have been created. The cause of this observed baryon asymmetry is not known. The distribution of matter throughout the universe is highly variable. The average density is about 1 proton per 200 litres. Vast volumes of the universe are voids of exceptionally low density. The interstellar medium far from stars but within a galaxy has density of a few protons per litre. The proportions of all types of matter and energy have changed over the history of the universe. The total amount of electromagnetic radiation generated within the universe has decreased by 1/2 in the past 2 billion years. Today, ordinary matter, which includes atoms, stars, galaxies, and life, accounts for only 4.9% of the contents of the universe. The present overall density of this type of matter is very low, roughly 4.5 × 10−31 grams per cubic centimeter, corresponding to a density of the order of only one proton for every four cubic meters of volume. The nature of both dark energy and dark matter is unknown. Dark matter, a mysterious form of matter that has not yet been identified, accounts for 26.8% of the cosmic contents. Dark energy, which is the energy of empty space and is causing the expansion of the universe to accelerate, accounts for the remaining 68.3% of the contents. 

Matter, dark matter, and dark energy are distributed homogeneously throughout the universe over length scales longer than 300 million light-years (ly) or so. However, over shorter length-scales, matter tends to clump hierarchically; many atoms are condensed into stars, most stars into galaxies, most galaxies into clusters, superclusters and, finally, large-scale galactic filaments. The observable universe contains as many as an estimated 2 trillion galaxies and, overall, as many as an estimated 1024 stars – more stars (and Earth-like planets) than all the grains of beach sand on planet Earth; but less than the total number of atoms estimated in the universe as 1082; and the estimated total number of stars in an inflationary universe (observed and unobserved), as 10100. Typical galaxies range from dwarfs with as few as ten million (107) stars up to giants with one trillion (1012) stars. Between the larger structures are voids, which are typically 10–150 Mpc (33 million–490 million ly) in diameter. The Milky Way is in the Local Group of galaxies, which in turn is in the Laniakea Supercluster. This supercluster spans over 500 million light-years, while the Local Group spans over 10 million light-years. The universe also has vast regions of relative emptiness; the largest known void measures 1.8 billion ly (550 Mpc) across. 

The observable universe is isotropic on scales significantly larger than superclusters, meaning that the statistical properties of the universe are the same in all directions as observed from Earth. The universe is bathed in highly isotropic microwave radiation that corresponds to a thermal equilibrium blackbody spectrum of roughly 2.72548 kelvins. The hypothesis that the large-scale universe is homogeneous and isotropic is known as the cosmological principle. A universe that is both homogeneous and isotropic looks the same from all vantage points and has no center. 

Dark energy 

An explanation for why the expansion of the universe is accelerating remains elusive. It is often attributed to the gravitational influence of "dark energy", an unknown form of energy that is hypothesized to permeate space. On a mass–energy equivalence basis, the density of dark energy (~ 7 × 10−30 g/cm3) is much less than the density of ordinary matter or dark matter within galaxies. However, in the present dark-energy era, it dominates the mass–energy of the universe because it is uniform across space. Two proposed forms for dark energy are the cosmological constant, a constant energy density filling space homogeneously, and scalar fields such as quintessence or moduli, dynamic quantities whose energy density can vary in time and space while still permeating them enough to cause the observed rate of expansion. Contributions from scalar fields that are constant in space are usually also included in the cosmological constant. The cosmological constant can be formulated to be equivalent to vacuum energy. 

Dark matter 

Dark matter is a hypothetical kind of matter that is invisible to the entire electromagnetic spectrum, but which accounts for most of the matter in the universe. The existence and properties of dark matter are inferred from its gravitational effects on visible matter, radiation, and the large-scale structure of the universe. Other than neutrinos, a form of hot dark matter, dark matter has not been detected directly, making it one of the greatest mysteries in modern astrophysics. Dark matter neither emits nor absorbs light or any other electromagnetic radiation at any significant level. Dark matter is estimated to constitute 26.8% of the total mass–energy and 84.5% of the total matter in the universe. 

Ordinary matter 

The remaining 4.9% of the mass–energy of the universe is ordinary matter, that is, atoms, ions, electrons and the objects they form. This matter includes stars, which produce nearly all of the light we see from galaxies, as well as interstellar gas in the interstellar and intergalactic media, planets, and all the objects from everyday life that we can bump into, touch or squeeze. The great majority of ordinary matter in the universe is unseen, since visible stars and gas inside galaxies and clusters account for less than 10 percent of the ordinary matter contribution to the mass–energy density of the universe. Ordinary matter commonly exists in four states (or phases): solid, liquid, gas, and plasma. However, advances in experimental techniques have revealed other previously theoretical phases, such as Bose–Einstein condensates and fermionic condensates. Ordinary matter is composed of two types of elementary particles: quarks and leptons. For example, the proton is formed of two up quarks and one down quark; the neutron is formed of two down quarks and one up quark; and the electron is a kind of lepton. An atom consists of an atomic nucleus, made up of protons and neutrons (both of which are baryons), and electrons that orbit the nucleus. Soon after the Big Bang, primordial protons and neutrons formed from the quark–gluon plasma of the early universe as it cooled below two trillion degrees. A few minutes later, in a process known as Big Bang nucleosynthesis, nuclei formed from the primordial protons and neutrons. This nucleosynthesis formed lighter elements, those with small atomic numbers up to lithium and beryllium, but the abundance of heavier elements dropped off sharply with increasing atomic number. Some boron may have been formed at this time, but the next heavier element, carbon, was not formed in significant amounts. Big Bang nucleosynthesis shut down after about 20 minutes due to the rapid drop in temperature and density of the expanding universe. Subsequent formation of heavier elements resulted from stellar nucleosynthesis and supernova nucleosynthesis. 

Cosmological models 

Model of the universe based on general relativity 

General relativity is the geometric theory of gravitation formulated by Albert Einstein in 1915 and remains the standard description of gravity in modern physics. It extends special relativity and Newton's law of universal gravitation by describing gravity as a manifestation of the curvature of space and time (spacetime). In this framework, the curvature of spacetime is determined by the energy and momentum of matter and radiation. This relationship is expressed through the Einstein field equations, which link the distribution of matter and energy to the geometry of spacetime. The resulting geometry governs the motion of matter, so that solutions of these equations describe how the universe evolves over time. Under the cosmological principle, which assumes that the universe is homogeneous and isotropic on large scales, the field equations admit a class of solutions described by the metric tensor known as the Friedmann–Lemaître–Robertson–Walker metric. In this description, the universe is characterized by two quantities: a scale factor, which describes how its overall size changes with time, and a curvature index, which specifies its spatial geometry. The curvature can be flat, positively curved, or negatively curved. The evolution of the scale factor depends on both the spatial curvature and the cosmological constant, which represents the energy density of empty space and may be associated with dark energy. The relation governing this evolution is known as the Friedmann equation, introduced by Alexander Friedmann. The curvature determines the global geometry of space. A positively curved universe has a finite volume and can be visualized as a three-dimensional sphere. A flat or negatively curved universe is spatially infinite. Although this may seem counterintuitive, models with flat or negative curvature allow an infinite universe to emerge from an initial state in which the scale factor vanishes, consistent with the cosmological principle. Analogies include an infinite plane (flat) or other geometries such as a torus. The ultimate fate of the universe depends on both the curvature and the cosmological constant. A sufficiently dense universe with positive curvature would eventually recollapse in a Big Crunch, possibly followed by a Big Bounce. In contrast, a flat or negatively curved universe would expand indefinitely, approaching a Big Freeze and eventual heat death of the universe. Observations indicate that the expansion of the universe is accelerating, raising the possibility of a Big Rip. Current data suggest that the universe is close to flat, with a density near the critical value separating recollapse from eternal expansion. 

Multiverse hypotheses 

Some speculative theories have proposed that our universe is but one of a set of disconnected universes, collectively denoted as the multiverse. An easily visualized metaphor of these concepts is a group of separate soap bubbles, in which observers living on one soap bubble cannot interact with those on other soap bubbles, even in principle. According to one common terminology, each "soap bubble" of spacetime is denoted as a universe, whereas humans' particular spacetime is denoted as the universe, just as humans call Earth's moon the Moon. The entire collection of these separate spacetimes is denoted as the multiverse. Max Tegmark and Brian Greene have proposed different classification schemes for multiverse ideas. In Tegmark's scheme multiverses might result from the immense size of the spacetime, from cosmological processes that produce spacetime bubbles, from quantum mechanical unitarity, or because we live in a mathematical construct. If space is infinite, or sufficiently large and uniform, identical instances of the history of Earth's entire Hubble volume occur every so often, simply by chance. Tegmark calculated that our nearest so-called doppelgänger is 1010115 metres away from us (a double exponential function larger than a googolplex). The physical basis of these ideas have been challenged. 

Historical conceptions 

Historically, there have been many ideas of the cosmos (cosmologies) and its origin (cosmogonies). Theories of an impersonal universe governed by physical laws were first proposed by the Greeks and Indians. Ancient Chinese philosophy encompassed the notion of the universe including both all of space and all of time. Over the centuries, improvements in astronomical observations and theories of motion and gravitation led to ever more accurate descriptions of the universe. The modern era of cosmology began with Albert Einstein's 1915 general theory of relativity, which made it possible to quantitatively predict the origin, evolution, and conclusion of the universe as a whole. Most modern, accepted theories of cosmology are based on general relativity and, more specifically, the predicted Big Bang. 

Mythologies 

Many cultures have stories describing the origin of the world and universe. Cultures generally regard these stories as having some truth. There are however many differing beliefs in how these stories apply amongst those believing in a supernatural origin, ranging from a god directly creating the universe as it is now to a god just setting the "wheels in motion" (for example via mechanisms such as the big bang and evolution). Ethnologists and anthropologists who study myths have developed various classification schemes for the various themes that appear in creation stories. For example, in one type of story, the world is born from a world egg; such stories include the Finnish epic poem Kalevala, the Chinese story of Pangu or the Indian Brahmanda Purana. In related stories, the universe is created by a single entity emanating or producing something by him- or herself, as in the Tibetan Buddhism concept of Adi-Buddha, the ancient Greek story of Gaia (Mother Earth), the Aztec goddess Coatlicue myth, the ancient Egyptian god Atum story, and the Judeo-Christian Genesis creation narrative in which the Abrahamic God created the universe. In another type of story, the universe is created from the union of male and female deities, as in the Māori story of Rangi and Papa. In other stories, the universe is created by crafting it from pre-existing materials, such as the corpse of a dead god—as from Tiamat in the Babylonian epic Enuma Elish or from the giant Ymir in Norse mythology—or from chaotic materials, as in Izanagi and Izanami in Japanese mythology. In other stories, the universe emanates from fundamental principles, such as Brahman and Prakrti, and the creation myth of the Serers. 

Philosophical models 

The pre-Socratic Greek philosophers and Indian philosophers developed some of the earliest philosophical concepts of the universe. The earliest Greek philosophers noted that appearances can be deceiving, and sought to understand the underlying reality behind the appearances. In particular, they noted the ability of matter to change forms (e.g., ice to water to steam) and several philosophers proposed that all the physical materials in the world are different forms of a single primordial material, or arche. The first to do so was Thales, who proposed this material to be water. Thales' student, Anaximander, proposed that everything came from the limitless apeiron. Anaximenes proposed the primordial material to be air on account of its perceived attractive and repulsive qualities that cause the arche to condense or dissociate into different forms. Anaxagoras proposed the principle of Nous (Mind), while Heraclitus proposed fire (and spoke of logos). Empedocles proposed the elements to be earth, water, air and fire. His four-element model became very popular. Like Pythagoras, Plato believed that all things were composed of number, with Empedocles' elements taking the form of the Platonic solids. Democritus, and later philosophers—most notably Leucippus—proposed that the universe is composed of indivisible atoms moving through a void (vacuum), although Aristotle did not believe that to be feasible because air, like water, offers resistance to motion. Air will immediately rush in to fill a void, and moreover, without resistance, it would do so indefinitely fast. Although Heraclitus argued for eternal change, his contemporary Parmenides emphasized changelessness. Parmenides' poem On Nature has been read as saying that all change is an illusion, that the true underlying reality is eternally unchanging and of a single nature, or at least that the essential feature of each thing that exists must exist eternally, without origin, change, or end. His student Zeno of Elea challenged everyday ideas about motion with several famous paradoxes. Aristotle responded to these paradoxes by developing the notion of a potential countable infinity, as well as the infinitely divisible continuum. The Indian philosopher Kanada, founder of the Vaisheshika school, developed a notion of atomism and proposed that light and heat were varieties of the same substance. In the 5th century AD, the Buddhist atomist philosopher Dignāga proposed atoms to be point-sized, durationless, and made of energy. They denied the existence of substantial matter and proposed that movement consisted of momentary flashes of a stream of energy. The notion of temporal finitism was inspired by the doctrine of creation shared by the three Abrahamic religions: Judaism, Christianity and Islam. The Christian philosopher, John Philoponus, presented the philosophical arguments against the ancient Greek notion of an infinite past and future. Philoponus' arguments against an infinite past were used by the early Muslim philosopher, Al-Kindi (Alkindus); the Jewish philosopher, Saadia Gaon (Saadia ben Joseph); and the Muslim theologian, Al-Ghazali (Algazel). Pantheism is the philosophical religious belief that the universe itself is identical to divinity and a supreme being or entity. The physical universe is thus understood as an all-encompassing, immanent deity. The term 'pantheist' designates one who holds both that everything constitutes a unity and that this unity is divine, consisting of an all-encompassing, manifested god or goddess. 

Astronomical concepts 

The earliest written records of identifiable predecessors to modern astronomy come from Ancient Egypt and Mesopotamia from around 3000 to 1200 BCE. Babylonian astronomers of the 7th century BCE viewed the world as a flat disk surrounded by the ocean. Later Greek philosophers, observing the motions of the heavenly bodies, were concerned with developing models of the universe based more profoundly on empirical evidence. Some of the earliest cosmological models of the universe were developed by ancient Greek and Indian philosophers and were geocentric, placing Earth at the center. The first coherent model was proposed by Eudoxus of Cnidos, a student of Plato who followed Plato's idea that heavenly motions had to be circular. In order to account for the known complications of the planets' motions, particularly retrograde movement, Eudoxus' model included 27 different celestial spheres: four for each of the planets visible to the naked eye, three each for the Sun and the Moon, and one for the stars. All of these spheres were centered on the Earth, which remained motionless while they rotated eternally. Aristotle elaborated upon this model, increasing the number of spheres to 55 in order to account for further details of planetary motion. For Aristotle, normal matter was entirely contained within the terrestrial sphere, and it obeyed fundamentally different rules from heavenly material. The post-Aristotle treatise De Mundo (of uncertain authorship and date) stated, "Five elements, situated in spheres in five regions, the less being in each case surrounded by the greater—namely, earth surrounded by water, water by air, air by fire, and fire by ether—make up the whole universe". This model was also refined by Callippus and after concentric spheres were abandoned, it was brought into nearly perfect agreement with astronomical observations by Ptolemy. The success of such a model is largely due to the mathematical fact that any function (such as the position of a planet) can be decomposed into a set of circular functions (the Fourier modes). Other Greek scientists, such as the Pythagorean philosopher Philolaus, postulated (according to Stobaeus' account) that at the center of the universe was a "central fire" around which the Earth, Sun, Moon and planets revolved in uniform circular motion. The Greek astronomer Aristarchus of Samos was the first known individual to propose a heliocentric model of the universe. Though the original text has been lost, a reference in Archimedes' book The Sand Reckoner describes Aristarchus's heliocentric model. Archimedes wrote: 

You, King Gelon, are aware the universe is the name given by most astronomers to the sphere the center of which is the center of the Earth, while its radius is equal to the straight line between the center of the Sun and the center of the Earth. This is the common account as you have heard from astronomers. But Aristarchus has brought out a book consisting of certain hypotheses, wherein it appears, as a consequence of the assumptions made, that the universe is many times greater than the universe just mentioned. His hypotheses are that the fixed stars and the Sun remain unmoved, that the Earth revolves about the Sun on the circumference of a circle, the Sun lying in the middle of the orbit, and that the sphere of fixed stars, situated about the same center as the Sun, is so great that the circle in which he supposes the Earth to revolve bears such a proportion to the distance of the fixed stars as the center of the sphere bears to its surface. Aristarchus thus believed the stars to be very far away, and saw this as the reason why stellar parallax had not been observed, that is, the stars had not been observed to move relative each other as the Earth moved around the Sun. The stars are in fact much farther away than the distance that was generally assumed in ancient times, which is why stellar parallax is only detectable with precision instruments. The geocentric model, consistent with planetary parallax, was assumed to be the explanation for the unobservability of stellar parallax. 

The only other astronomer from antiquity known by name who supported Aristarchus's heliocentric model was Seleucus of Seleucia, a Hellenistic astronomer who lived a century after Aristarchus. According to Plutarch, Seleucus was the first to prove the heliocentric system through reasoning, but it is not known what arguments he used. Seleucus' arguments for a heliocentric cosmology were probably related to the phenomenon of tides. According to Strabo (1.1.9), Seleucus was the first to state that the tides are due to the attraction of the Moon, and that the height of the tides depends on the Moon's position relative to the Sun. Alternatively, he may have proved heliocentricity by determining the constants of a geometric model for it, and by developing methods to compute planetary positions using this model, similar to Nicolaus Copernicus in the 16th century. During the Middle Ages, heliocentric models were also proposed by the Persian astronomers Albumasar and Al-Sijzi. 

The Aristotelian model was accepted in the Western world for roughly two millennia, until Copernicus revived Aristarchus's perspective that the astronomical data could be explained more plausibly if the Earth rotated on its axis and if the Sun were placed at the center of the universe. 

In the center rests the Sun. For who would place this lamp of a very beautiful temple in another or better place than this wherefrom it can illuminate everything at the same time? As noted by Copernicus, the notion that the Earth rotates is very old, dating at least to Philolaus (c. 450 BC), Heraclides Ponticus (c. 350 BC) and Ecphantus the Pythagorean. Roughly a century before Copernicus, the Christian scholar Nicholas of Cusa also proposed that the Earth rotates on its axis in his book, On Learned Ignorance (1440). Al-Sijzi also proposed that the Earth rotates on its axis. Empirical evidence for the Earth's rotation on its axis, using the phenomenon of comets, was given by Tusi (1201–1274) and Ali Qushji (1403–1474). This cosmology was accepted by Isaac Newton, Christiaan Huygens and later scientists. Newton demonstrated that the same laws of motion and gravity apply to earthly and to celestial matter, making Aristotle's division between the two obsolete. Edmund Halley (1720) and Jean-Philippe de Chéseaux (1744) noted independently that the assumption of an infinite space filled uniformly with stars would lead to the prediction that the nighttime sky would be as bright as the Sun itself; this became known as Olbers' paradox in the 19th century. Newton believed that an infinite space uniformly filled with matter would cause infinite forces and instabilities causing the matter to be crushed inwards under its own gravity. This instability was clarified in 1902 by the Jeans instability criterion. One solution to these paradoxes is the Charlier universe, in which the matter is arranged hierarchically (systems of orbiting bodies that are themselves orbiting in a larger system, ad infinitum) in a fractal way such that the universe has a negligibly small overall density; such a cosmological model had also been proposed earlier in 1761 by Johann Heinrich Lambert. 

Deep space astronomy During the 18th century, Immanuel Kant speculated that nebulae could be entire galaxies separate from the Milky Way, and in 1850, Alexander von Humboldt called these separate galaxies Weltinseln, or "world islands", a term that later developed into "island universes". In 1919, when the Hooker Telescope was completed, the prevailing view was that the universe consisted entirely of the Milky Way Galaxy. Using the Hooker Telescope, Edwin Hubble identified Cepheid variables in several spiral nebulae and in 1922–1923 proved conclusively that Andromeda Nebula and Triangulum among others, were entire galaxies outside our own, thus proving that the universe consists of a multitude of galaxies. With this Hubble formulated the Hubble constant, which allowed for the first time a calculation of the age of the universe and size of the observable Universe, which became increasingly precise with better measurements, starting at 2 billion years and 280 million light-years, until 2006 when data of the Hubble Space Telescope allowed a very accurate calculation of the age of the universe and size of the Observable Universe. The modern era of physical cosmology began in 1917, when Albert Einstein first applied his general theory of relativity to model the structure and dynamics of the universe. The discoveries of this era, and the questions that remain unanswered, are outlined in the sections above. 

See also 

References Footnotes 

Citations 

Bibliography Van Der Waerden, B. L. (June 1987). "The Heliocentric System in Greek, Persian and Hindu Astronomy". Annals of the New York Academy of Sciences. 500 (1): 525–545. Bibcode:1987NYASA.500..525V. doi:10.1111/j.1749-6632.1987.tb37224.x. ISSN 0077-8923. S2CID 222087224. Landau LD, Lifshitz EM (1975). The classical theory of fields. Course of theoretical physics. Vol. 2 (4th rev. English ed.). Oxford; New York: Pergamon Press. pp. 358–397. ISBN 978-0-08-018176-9. Liddell, Henry George & Scott, Robert (1994). A Greek-English lexicon. Oxford: Clarendon Pr. ISBN 978-0-19-864214-5. Misner, Charles W.; Thorne, Kip S.; Wheeler, John Archibald; Kip; Wheeler; J.A. (2008). Gravitation (27. printing ed.). New York, NY: Freeman. pp. 703–816. ISBN 978-0-7167-0344-0. Raine, Derek; Thomas, Edwin G. (2001). An introduction to the science of cosmology. Series in astronomy and astrophysics. Bristol: Institute of Physics Publ. ISBN 978-0-7503-0405-4. Rindler, Wolfgang (1986). Essential relativity: special, general, and cosmological. Texts and monographs in physics. New York Heidelberg: Springer. pp. 193–244. ISBN 978-0-387-10090-6. Rees, Martin J.; DK Publishing, Inc; Smithsonian Institution, eds. (2012). Universe (Rev. ed.). New York: DK Pub. ISBN 978-0-7566-9841-6. OCLC 809932784. 

External links 

NASA/IPAC Extragalactic Database (NED) / (NED-Distances). There are about 1082 atoms in the observable universe – LiveScience, July 2021. This is why we will never know everything about our universe – Forbes, May 2019. 

[Language] Language is a structured system of communication that consists of grammar and vocabulary. It is the primary means by which humans convey meaning, both in spoken and signed forms, and may also be conveyed through writing. Human language is characterized by its cultural and historical diversity, with significant variations observed between cultures and across time. Human languages possess the properties of productivity and displacement, which enable the creation of an infinite number of sentences, and the ability to refer to objects, events, and ideas that are not immediately present in the discourse. The use of human language relies on social convention and is acquired through learning. Estimates of the number of human languages in the world vary between 5,000 and 7,000. Precise estimates depend on an arbitrary distinction (dichotomy) established between languages and dialects. Natural languages are spoken, signed, or both; however, any language can be encoded into secondary media using auditory, visual, or tactile stimuli – for example, writing, whistling, signing, or braille. In other words, human language is modality-independent, but written or signed language is the way to inscribe or encode the natural human speech or gestures. Depending on philosophical perspectives regarding the definition of language and meaning, when used as a general concept, "language" may refer to the cognitive ability to learn and use systems of complex communication, or to describe the set of rules that makes up these systems, or the set of utterances that can be produced from those rules. All languages rely on the process of semiosis to relate signs to particular meanings. Oral, manual and tactile languages contain a phonological system that governs how symbols are used to form sequences known as words or morphemes, and a syntactic system that governs how words and morphemes are combined to form phrases and utterances. The scientific study of language is called linguistics. Critical examinations of languages, such as philosophy of language, the relationships between language and thought, how words represent experience, etc., have been debated at least since Gorgias and Plato in ancient Greek civilization. Thinkers such as Jean-Jacques Rousseau (1712–1778) have argued that language originated from emotions, while others like Immanuel Kant (1724–1804) have argued that languages originated from rational and logical thought. Twentieth century philosophers such as Ludwig Wittgenstein (1889–1951) argued that philosophy is really the study of language itself. Major figures in contemporary linguistics include Ferdinand de Saussure and Noam Chomsky. Language is thought to have gradually diverged from earlier primate communication systems when early hominins acquired the ability to form a theory of mind and shared intentionality. This development is sometimes thought to have coincided with an increase in brain volume, and many linguists see the structures of language as having evolved to serve specific communicative and social functions. Language is processed in many different locations in the human brain, but especially in Broca's and Wernicke's areas. Humans acquire language through social interaction in early childhood, and children generally speak fluently by approximately three years old. Language and culture are codependent. Therefore, in addition to its strictly communicative uses, language has social uses such as signifying group identity, social stratification, as well as use for social grooming and entertainment. Languages evolve and diversify over time, and the history of their evolution can be reconstructed by comparing modern languages to determine which traits their ancestral languages must have had in order for the later developmental stages to occur. A group of languages that descend from a common ancestor is known as a language family; in contrast, a language that has been demonstrated not to have any living or non-living relationship with another language is called a language isolate. There are also many unclassified languages whose relationships have not been established, and spurious languages may have not existed at all. Academic consensus holds that between 50% and 90% of languages spoken at the beginning of the 21st century will probably have become extinct by the year 2100. 

Definitions 

The English word language derives ultimately from Proto-Indo-European *dn̥ǵʰwéh₂s "tongue, speech, language" through Latin lingua, "language; tongue", and Old French language. The word is sometimes used to refer to codes, ciphers, and other kinds of artificially constructed communication systems such as formally defined computer languages used for computer programming. Unlike conventional human languages, a formal language in this sense is a system of signs for encoding and decoding information. This article specifically concerns the properties of natural human language as it is studied in the discipline of linguistics. As an object of linguistic study, "language" has two primary meanings: an abstract concept, and a specific linguistic system, e.g. "French". The Swiss linguist Ferdinand de Saussure, who defined the modern discipline of linguistics, first explicitly formulated the distinction using the French word langage for language as a concept, langue as a specific instance of a language system, and parole for the concrete use of speech in a particular language. When speaking of language as a general concept, definitions can be used which stress different aspects of the phenomenon. These definitions also entail different approaches and understandings of language, and they also inform different and often incompatible schools of linguistic theory. Debates about the nature and origin of language go back to the ancient world. Greek philosophers such as Gorgias and Plato debated the relation between words, concepts and reality. Gorgias argued that language could represent neither the objective experience nor human experience, and that communication and truth were therefore impossible. Plato maintained that communication is possible because language represents ideas and concepts that exist independently of, and prior to, language. During the Enlightenment and its debates about human origins, it became fashionable to speculate about the origin of language. Thinkers such as Rousseau and Johann Gottfried Herder argued that language had originated in the instinctive expression of emotions, and that it was originally closer to music and poetry than to the logical expression of rational thought. Rationalist philosophers such as Kant and René Descartes held the opposite view. Around the turn of the 20th century, thinkers began to wonder about the role of language in shaping our experiences of the world – asking whether language simply reflects the objective structure of the world, or whether it creates concepts that in turn impose structure on our experience of the objective world. This led to the question of whether philosophical problems are really firstly linguistic problems. The resurgence of the view that language plays a significant role in the creation and circulation of concepts, and that the study of philosophy is essentially the study of language, is associated with what has been called the linguistic turn and philosophers such as Wittgenstein in 20th-century philosophy. These debates about language in relation to meaning and reference, cognition and consciousness remain active today. 

Mental faculty, organ or instinct One definition sees language primarily as the mental faculty that allows humans to undertake linguistic behaviour: to learn languages and to produce and understand utterances. This definition stresses the universality of language to all humans, and it emphasizes the biological basis for the human capacity for language as a unique development of the human brain. Proponents of the view that the drive to language acquisition is innate in humans argue that this is supported by the fact that all cognitively normal children raised in an environment where language is accessible will acquire language without formal instruction. Languages may even develop spontaneously in environments where people live or grow up together without a common language; for example, creole languages and spontaneously developed sign languages such as Nicaraguan Sign Language. This view, which can be traced back to the philosophers Kant and Descartes, understands language to be largely innate, for example, in Chomsky's theory of universal grammar, or American philosopher Jerry Fodor's extreme innatist theory. These kinds of definitions are often applied in studies of language within a cognitive science framework and in neurolinguistics. 

Formal symbolic system Another definition sees language as a formal system of signs governed by grammatical rules of combination to communicate meaning. This definition stresses that human languages can be described as closed structural systems consisting of rules that relate particular signs to particular meanings. This structuralist view of language was first introduced by Ferdinand de Saussure, and his structuralism remains foundational for many approaches to language. Some proponents of Saussure's view of language have advocated a formal approach that studies language structure by identifying its basic elements and then by presenting a formal account of the rules according to which the elements combine in order to form words and sentences. The main proponent of such a theory is Noam Chomsky, the originator of the generative theory of grammar, who has defined language as the construction of sentences that can be generated using transformational grammars. Chomsky considers these rules to be an innate feature of the human mind and to constitute the rudiments of what language is. By way of contrast, such transformational grammars are also commonly used in formal logic, in formal linguistics, and in applied computational linguistics. In the philosophy of language, the view of linguistic meaning as residing in the logical relations between propositions and reality was developed by philosophers such as Alfred Tarski, Bertrand Russell, and other formal logicians. 

Tool for communication 

Yet another definition sees language as a system of communication that enables humans to exchange verbal or symbolic utterances. This definition stresses the social functions of language and the fact that humans use it to express themselves and to manipulate objects in their environment. Functional theories of grammar explain grammatical structures by their communicative functions, and understand the grammatical structures of language to be the result of an adaptive process by which grammar was "tailored" to serve the communicative needs of its users. This view of language is associated with the study of language in pragmatic, cognitive, and interactive frameworks, as well as in sociolinguistics and linguistic anthropology. Functionalist theories tend to study grammar as dynamic phenomena, as structures that are always in the process of changing as they are employed by their speakers. This view places importance on the study of linguistic typology, or the classification of languages according to structural features, as processes of grammaticalization tend to follow trajectories that are partly dependent on typology. In the philosophy of language, the view of pragmatics as being central to language and meaning is often associated with Wittgenstein's later works and with ordinary language philosophers such as J. L. Austin, Paul Grice, John Searle, and W.O. Quine. 

Human versus animal language 

A number of features, many of which were described by Charles Hockett and called design features set human language apart from communication used by non-human animals. Communication systems used by other animals such as bees or apes are closed systems that consist of a finite, usually very limited, number of possible ideas that can be expressed. In contrast, human language is open-ended and productive, meaning that it allows humans to produce a vast range of utterances from a finite set of elements, and to create new words and sentences. This is possible because human language is based on a dual code, in which a finite number of elements which are meaningless in themselves (e.g. sounds, letters or gestures) can be combined to form an infinite number of larger units of meaning (words and sentences). However, one study has demonstrated that an Australian bird, the chestnut-crowned babbler, is capable of using the same acoustic elements in different arrangements to create two functionally distinct vocalizations. Additionally, pied babblers have demonstrated the ability to generate two functionally distinct vocalisations composed of the same sound type, which can only be distinguished by the number of repeated elements. Several species of animals have proved to be able to acquire forms of communication through social learning: for instance a bonobo named Kanzi learned to express itself using a set of symbolic lexigrams. Similarly, many species of birds and whales learn their songs by imitating other members of their species. However, while some animals may acquire large numbers of words and symbols, none have been able to learn as many different signs as are generally known by an average 4 year old human, nor have any acquired anything resembling the complex grammar of human language. Human languages differ from animal communication systems in that they employ grammatical and semantic categories, such as noun and verb, present and past, which may be used to express exceedingly complex meanings. It is distinguished by the property of recursivity: for example, a noun phrase can contain another noun phrase (as in "[[the chimpanzee]'s lips]") or a clause can contain another clause (as in "[I see [the dog is running]]"). Human language is the only known natural communication system whose adaptability may be referred to as modality independent. This means that it can be used not only for communication through one channel or medium, but through several. For example, spoken language uses the auditive modality, whereas sign languages and writing use the visual modality, and braille writing uses the tactile modality. Human language is unusual in being able to refer to abstract concepts and to imagined or hypothetical events as well as events that took place in the past or may happen in the future. This ability to refer to events that are not at the same time or place as the speech event is called displacement, and while some animal communication systems can use displacement (such as the communication of bees that can communicate the location of sources of nectar that are out of sight), the degree to which it is used in human language is also considered unique. 

Origin 

Theories about the origin of language differ in regard to their basic assumptions about what language is. Some theories are based on the idea that language is so complex that one cannot imagine it simply appearing from nothing in its final form, but that it must have evolved from earlier pre-linguistic systems among our pre-human ancestors. These theories can be called continuity-based theories. The opposite viewpoint is that language is such a unique human trait that it cannot be compared to anything found among non-humans and that it must therefore have appeared suddenly in the transition from pre-hominids to early man. These theories can be defined as discontinuity-based. Similarly, theories based on the generative view of language pioneered by Noam Chomsky see language mostly as an innate faculty that is largely genetically encoded, whereas functionalist theories see it as a system that is largely cultural, learned through social interaction. 

Continuity-based theories are held by a majority of scholars, but they vary in how they envision this development. Those who see language as being mostly innate, such as psychologist Steven Pinker, hold the precedents to be animal cognition, whereas those who see language as a socially learned tool of communication, such as psychologist Michael Tomasello, see it as having developed from animal communication in primates: either gestural or vocal communication to assist in cooperation. Other continuity-based models see language as having developed from music, a view already espoused by Rousseau, Herder, Humboldt, and Charles Darwin. A prominent proponent of this view is archaeologist Steven Mithen. Stephen Anderson states that the age of spoken languages is estimated at 60,000 to 100,000 years and that: Researchers on the evolutionary origin of language generally find it plausible to suggest that language was invented only once, and that all modern spoken languages are thus in some way related, even if that relation can no longer be recovered ... because of limitations on the methods available for reconstruction. Because language emerged in the early prehistory of man, before the existence of any written records, its early development has left no historical traces, and it is believed that no comparable processes can be observed today. Theories that stress continuity often look at animals to see if, for example, primates display any traits that can be seen as analogous to what pre-human language must have been like. Early human fossils can be inspected for traces of physical adaptation to language use or pre-linguistic forms of symbolic behaviour. Among the signs in human fossils that may suggest linguistic abilities are: the size of the brain relative to body mass, the presence of a larynx capable of advanced sound production and the nature of tools and other manufactured artifacts. It was mostly undisputed that pre-human australopithecines did not have communication systems significantly different from those found in great apes in general. However, a 2017 study on Ardipithecus ramidus challenges this belief. Scholarly opinions vary as to the developments since the appearance of the genus Homo some 2.5 million years ago. Some scholars assume the development of primitive language-like systems (proto-language) as early as Homo habilis (2.3 million years ago) while others place the development of primitive symbolic communication only with Homo erectus (1.8 million years ago) or Homo heidelbergensis (0.6 million years ago), and the development of language proper with anatomically modern Homo sapiens with the Upper Paleolithic revolution less than 100,000 years ago. Chomsky is one prominent proponent of a discontinuity-based theory of human language origins. He suggests that for scholars interested in the nature of language, "talk about the evolution of the language capacity is beside the point." Chomsky proposes that perhaps "some random mutation took place [...] and it reorganized the brain, implanting a language organ in an otherwise primate brain." Though cautioning against taking this story literally, Chomsky insists that "it may be closer to reality than many other fairy tales that are told about evolutionary processes, including language." In March 2024, researchers reported that the beginnings of human language began about 1.6 million years ago. 

Study 

The study of language, linguistics, has been developing into a science since the first grammatical descriptions of particular languages in India more than 2000 years ago, after the development of the Brahmi script. Modern linguistics is a science that concerns itself with all aspects of language, examining it from all of the theoretical viewpoints described above. 

Subdisciplines The academic study of language is conducted within many different disciplinary areas and from different theoretical angles, all of which inform modern approaches to linguistics. For example, descriptive linguistics examines the grammar of single languages, theoretical linguistics develops theories on how best to conceptualize and define the nature of language based on data from the various extant human languages, sociolinguistics studies how languages are used for social purposes informing in turn the study of the social functions of language and grammatical description, neurolinguistics studies how language is processed in the human brain and allows the experimental testing of theories, computational linguistics builds on theoretical and descriptive linguistics to construct computational models of language often aimed at processing natural language or at testing linguistic hypotheses, and historical linguistics relies on grammatical and lexical descriptions of languages to trace their individual histories and reconstruct trees of language families by using the comparative method. 

Early history 

The formal study of language is often considered to have started in India with Pāṇini, the 5th century BC grammarian who formulated 3,959 rules of Sanskrit morphology. However, Sumerian scribes already studied the differences between Sumerian and Akkadian grammar around 1900 BC. Subsequent grammatical traditions developed in all of the ancient cultures that adopted writing. In the 17th century AD, the French Port-Royal Grammarians developed the idea that the grammars of all languages were a reflection of the universal basics of thought, and therefore that grammar was universal. In the 18th century, the first use of the comparative method by British philologist and expert on ancient India William Jones sparked the rise of comparative linguistics. The scientific study of language was broadened from Indo-European to language in general by Wilhelm von Humboldt. Early in the 20th century, Ferdinand de Saussure introduced the idea of language as a static system of interconnected units, defined through the oppositions between them. By introducing a distinction between diachronic and synchronic analyses of language, he laid the foundation of the modern discipline of linguistics. Saussure also introduced several basic dimensions of linguistic analysis that are still fundamental in many contemporary linguistic theories, such as the distinctions between syntagm and paradigm, and the Langue-parole distinction, distinguishing language as an abstract system (langue), from language as a concrete manifestation of this system (parole). 

Modern linguistics 

In the 1960s, Noam Chomsky formulated the generative theory of language. According to this theory, the most basic form of language is a set of syntactic rules that is universal for all humans and which underlies the grammars of all human languages. This set of rules is called Universal Grammar; for Chomsky, describing it is the primary objective of the discipline of linguistics. Thus, he considered that the grammars of individual languages are only of importance to linguistics insofar as they allow us to deduce the universal underlying rules from which the observable linguistic variability is generated. In opposition to the formal theories of the generative school, functional theories of language propose that since language is fundamentally a tool, its structures are best analyzed and understood by reference to their functions. Formal theories of grammar seek to define the different elements of language and describe the way they relate to each other as systems of formal rules or operations, while functional theories seek to define the functions performed by language and then relate them to the linguistic elements that carry them out. The framework of cognitive linguistics interprets language in terms of the concepts (which are sometimes universal, and sometimes specific to a particular language) which underlie its forms. Cognitive linguistics is primarily concerned with how the mind creates meaning through language. 

Physiological and neural architecture of language and speech Speaking is the default modality for language in all cultures with hearing members. The production of spoken language depends on sophisticated capacities for controlling the lips, tongue and other components of the vocal apparatus, the ability to acoustically decode speech sounds, and the neurological apparatus required for acquiring and producing language. The study of the genetic bases for human language is at an early stage: the only gene that has definitely been implicated in language production is FOXP2, which may cause a kind of congenital language disorder if affected by mutations. 

The brain 

The brain is the coordinating center of all linguistic activity; it controls both the production of linguistic cognition and of meaning and the mechanics of speech production. Nonetheless, our knowledge of the neurological bases for language is quite limited, though it has advanced considerably with the use of modern imaging techniques. The discipline of linguistics dedicated to studying the neurological aspects of language is called neurolinguistics. Early work in neurolinguistics involved the study of language in people with brain lesions, to see how lesions in specific areas affect language and speech. In this way, neuroscientists in the 19th century discovered that two areas in the brain are crucially implicated in language processing. The first area is Wernicke's area, which is in the posterior section of the superior temporal gyrus in the dominant cerebral hemisphere. People with a lesion in this area of the brain develop receptive aphasia, a condition in which there is a major impairment of language comprehension, while speech retains a natural-sounding rhythm and a relatively normal sentence structure. The second area is Broca's area, in the posterior inferior frontal gyrus of the dominant hemisphere. People with a lesion to this area develop expressive aphasia, meaning that they know what they want to say, they just cannot get it out. They are typically able to understand what is being said to them, but unable to speak fluently. Other symptoms that may be present in expressive aphasia include problems with word repetition. The condition affects both spoken and written language. Those with this aphasia also exhibit ungrammatical speech and show inability to use syntactic information to determine the meaning of sentences. Both expressive and receptive aphasia also affect the use of sign language, in analogous ways to how they affect speech, with expressive aphasia causing signers to sign slowly and with incorrect grammar, whereas a signer with receptive aphasia will sign fluently, but make little sense to others and have difficulties comprehending others' signs. This shows that the impairment is specific to the ability to use language, not to the physiology used for speech production. With technological advances in the late 20th century, neurolinguists have also incorporated non-invasive techniques such as functional magnetic resonance imaging (fMRI) and electrophysiology to study language processing in individuals without impairments. 

Anatomy of speech 

Spoken language relies on human physical ability to produce sound, which is a longitudinal wave propagated through the air at a frequency capable of vibrating the ear drum. This ability depends on the physiology of the human speech organs. These organs consist of the lungs, the voice box (larynx), and the upper vocal tract – the throat, the mouth, and the nose. By controlling the different parts of the speech apparatus, the airstream can be manipulated to produce different speech sounds. The sound of speech can be analyzed into a combination of segmental and suprasegmental elements. The segmental elements are those that follow each other in sequences, which are usually represented by distinct letters in alphabetic scripts, such as the Roman script. In free flowing speech, there are no clear boundaries between one segment and the next, nor usually are there any audible pauses between them. Segments therefore are distinguished by their distinct sounds which are a result of their different articulations, and can be either vowels or consonants. Suprasegmental phenomena encompass such elements as stress, phonation type, voice timbre, and prosody or intonation, all of which may have effects across multiple segments. Consonants and vowel segments combine to form syllables, which in turn combine to form utterances; these can be distinguished phonetically as the space between two inhalations. Acoustically, these different segments are characterized by different formant structures, that are visible in a spectrogram of the recorded sound wave. Formants are the amplitude peaks in the frequency spectrum of a specific sound. Vowels are those sounds that have no audible friction caused by the narrowing or obstruction of some part of the upper vocal tract. They vary in quality according to the degree of lip aperture and the placement of the tongue within the oral cavity. Vowels are called close when the lips are relatively closed, as in the pronunciation of the vowel [i] (English "ee"), or open when the lips are relatively open, as in the vowel [a] (English "ah"). If the tongue is located towards the back of the mouth, the quality changes, creating vowels such as [u] (English "oo"). The quality also changes depending on whether the lips are rounded as opposed to unrounded, creating distinctions such as that between [i] (unrounded front vowel such as English "ee") and [y] (rounded front vowel such as German "ü"). Consonants are those sounds that have audible friction or closure at some point within the upper vocal tract. Consonant sounds vary by place of articulation, i.e. the place in the vocal tract where the airflow is obstructed, commonly at the lips, teeth, alveolar ridge, palate, velum, uvula, or glottis. Each place of articulation produces a different set of consonant sounds, which are further distinguished by manner of articulation, or the kind of friction, whether full closure, in which case the consonant is called occlusive or stop, or different degrees of aperture creating fricatives and approximants. Consonants can also be either voiced or unvoiced, depending on whether the vocal cords are set in vibration by airflow during the production of the sound. Voicing is what separates English [s] in bus (unvoiced sibilant) from [z] in buzz (voiced sibilant). Some speech sounds, both vowels and consonants, involve release of air flow through the nasal cavity, and these are called nasals or nasalized sounds. Other sounds are defined by the way the tongue moves within the mouth such as the l-sounds (called laterals, because the air flows along both sides of the tongue), and the r-sounds (called rhotics). By using these speech organs, humans can produce hundreds of distinct sounds: some appear very often in the world's languages, whereas others are much more common in certain language families, language areas, or even specific to a single language. 

Modality Human languages display considerable plasticity in their deployment of two fundamental modes: oral (speech and mouthing) and manual (sign and gesture). For example, it is common for oral language to be accompanied by gesture, and for sign language to be accompanied by mouthing. In addition, some language communities use both modes to convey lexical or grammatical meaning, each mode complementing the other. Such bimodal use of language is especially common in genres such as story-telling (with Plains Indian Sign Language and Australian Aboriginal sign languages used alongside oral language, for example), but also occurs in mundane conversation. For instance, many Australian languages have a rich set of case suffixes that provide details about the instrument used to perform an action. Others lack such grammatical precision in the oral mode, but supplement it with gesture to convey that information in the sign mode. In Iwaidja, for example, 'he went out for fish using a torch' is spoken as simply "he-hunted fish torch", but the word for 'torch' is accompanied by a gesture indicating that it was held. In another example, the ritual language Damin had a heavily reduced oral vocabulary of only a few hundred words, each of which was very general in meaning, but which were supplemented by gesture for greater precision (e.g., the single word for fish, l*i, was accompanied by a gesture to indicate the kind of fish). Secondary modes of language, by which a fundamental mode is conveyed in a different medium, include writing (including braille), sign (in manually coded language), whistling and drumming. Tertiary modes – such as semaphore, Morse code and spelling alphabets – convey the secondary mode of writing in a different medium. For some extinct languages that are maintained for ritual or liturgical purposes, writing may be the primary mode, with speech secondary. 

Structure When described as a system of symbolic communication, language is traditionally seen as consisting of three parts: signs, meanings, and a code connecting signs with their meanings. The study of the process of semiosis, how signs and meanings are combined, used, and interpreted is called semiotics. Signs can be composed of sounds, gestures, letters, or symbols, depending on whether the language is spoken, signed, or written, and they can be combined into complex signs, such as words and phrases. When used in communication, a sign is encoded and transmitted by a sender through a channel to a receiver who decodes it. 

Some of the properties that define human language as opposed to other communication systems are: the arbitrariness of the linguistic sign, meaning that there is no predictable connection between a linguistic sign and its meaning; the duality of the linguistic system, meaning that linguistic structures are built by combining elements into larger structures that can be seen as layered, e.g. how sounds build words and words build phrases; the discreteness of the elements of language, meaning that the elements out of which linguistic signs are constructed are discrete units, e.g. sounds and words, that can be distinguished from each other and rearranged in different patterns; and the productivity of the linguistic system, meaning that the finite number of linguistic elements can be combined into a theoretically infinite number of combinations. The rules by which signs can be combined to form words and phrases are called syntax or grammar. The meaning that is connected to individual signs, morphemes, words, phrases, and texts is called semantics. The division of language into separate but connected systems of sign and meaning goes back to the first linguistic studies of de Saussure and is now used in almost all branches of linguistics. 

Semantics 

Languages express meaning by relating a sign form to a meaning, or its content. Sign forms must be something that can be perceived, for example, in sounds, images, or gestures, and then related to a specific meaning by social convention. Because the basic relation of meaning for most linguistic signs is based on social convention, linguistic signs can be considered arbitrary, in the sense that the convention is established socially and historically, rather than by means of a natural relation between a specific sign form and its meaning. Thus, languages must have a vocabulary of signs related to specific meaning. The English sign "dog" denotes, for example, a member of the species Canis familiaris. In a language, the array of arbitrary signs connected to specific meanings is called the lexicon, and a single sign connected to a meaning is called a lexeme. Not all meanings in a language are represented by single words. Often, semantic concepts are embedded in the morphology or syntax of the language in the form of grammatical categories. All languages contain the semantic structure of predication: a structure that predicates a property, state, or action. Traditionally, semantics has been understood to be the study of how speakers and interpreters assign truth values to statements, so that meaning is understood to be the process by which a predicate can be said to be true or false about an entity, e.g. "[x [is y]]" or "[x [does y]]". Recently, this model of semantics has been complemented with more dynamic models of meaning that incorporate shared knowledge about the context in which a sign is interpreted into the production of meaning. Such models of meaning are explored in the field of pragmatics. 

Sounds and symbols 

Depending on modality, language structure can be based on systems of sounds (speech), gestures (sign languages), or graphic or tactile symbols (writing). The ways in which languages use sounds or signs to construct meaning are studied in phonology. Sounds as part of a linguistic system are called phonemes. Phonemes are abstract units of sound, defined as the smallest units in a language that can serve to distinguish between the meaning of a pair of minimally different words, a so-called minimal pair. In English, for example, the words bat [bæt] and pat [pʰæt] form a minimal pair, in which the distinction between /b/ and /p/ differentiates the two words, which have different meanings. However, each language contrasts sounds in different ways. For example, in a language that does not distinguish between voiced and unvoiced consonants, the sounds [p] and [b] (if they both occur) could be considered a single phoneme, and consequently, the two pronunciations would have the same meaning. Similarly, the English language does not distinguish phonemically between aspirated and non-aspirated pronunciations of consonants, as many other languages like Korean and Hindi do: the unaspirated /p/ in spin [spɪn] and the aspirated /p/ in pin [pʰɪn] are considered to be merely different ways of pronouncing the same phoneme (such variants of a single phoneme are called allophones), whereas in Mandarin Chinese, the same difference in pronunciation distinguishes between the words [pʰá] 'crouch' and [pā] 'eight' (the accent above the á means that the vowel is pronounced with a high tone and the accent above the ā means that the vowel is pronounced with a flat tone). All spoken languages have phonemes of at least two different categories, vowels and consonants, that can be combined to form syllables. As well as segments such as consonants and vowels, some languages also use sound in other ways to convey meaning. Many languages, for example, use stress, pitch, duration, and tone to distinguish meaning. Because these phenomena operate outside of the level of single segments, they are called suprasegmental. Some languages have only a few phonemes, for example, Rotokas and Pirahã language with 11 and 10 phonemes respectively, whereas languages like Taa may have as many as 141 phonemes. In sign languages, the equivalent to phonemes (formerly called cheremes) are defined by the basic elements of gestures, such as hand shape, orientation, location, and motion, which correspond to manners of articulation in spoken language. Writing systems represent language using visual symbols, which may or may not correspond to the sounds of spoken language. The Latin alphabet (and those on which it is based or that have been derived from it) was originally based on the representation of single sounds, so that words were constructed from letters that generally denote a single consonant or vowel in the structure of the word. In syllabic scripts, such as the Inuktitut syllabary, each sign represents a whole syllable. In logographic scripts, each sign represents an entire word, and will generally bear no relation to the sound of that word in spoken language. Because all languages have a very large number of words, no purely logographic scripts are known to exist. Written language represents the way spoken sounds and words follow one after another by arranging symbols according to a pattern that follows a certain direction. The direction used in a writing system is entirely arbitrary and established by convention. Some writing systems use the horizontal axis (left to right as the Latin script or right to left as the Arabic script), while others such as traditional Chinese writing use the vertical dimension (from top to bottom). A few writing systems use opposite directions for alternating lines, and others, such as the ancient Maya script, can be written in either direction and rely on graphic cues to show the reader the direction of reading. In order to represent the sounds of the world's languages in writing, linguists have developed the International Phonetic Alphabet, designed to represent all of the discrete sounds that are known to contribute to meaning in human languages. 

Grammar 

Grammar is the study of how meaningful elements called morphemes within a language can be combined into utterances. Morphemes can either be free or bound. If they are free to be moved around within an utterance, they are usually called words, and if they are bound to other words or morphemes, they are called affixes. The way in which meaningful elements can be combined within a language is governed by rules. The study of the rules for the internal structure of words are called morphology. The rules of the internal structure of phrases and sentences are called syntax. 

Grammatical categories 

Grammar can be described as a system of categories and a set of rules that determine how categories combine to form different aspects of meaning. Languages differ widely in whether they are encoded through the use of categories or lexical units. However, several categories are so common as to be nearly universal. Such universal categories include the encoding of the grammatical relations of participants and predicates by grammatically distinguishing between their relations to a predicate, the encoding of temporal and spatial relations on predicates, and a system of grammatical person governing reference to and distinction between speakers and addressees and those about whom they are speaking. 

Word classes Languages organize their parts of speech into classes according to their functions and positions relative to other parts. All languages, for instance, make a basic distinction between a group of words that prototypically denotes things and concepts and a group of words that prototypically denotes actions and events. The first group, which includes English words such as "dog" and "song", are usually called nouns. The second, which includes "think" and "sing", are called verbs. Another common category is the adjective: words that describe properties or qualities of nouns, such as "red" or "big". Word classes can be "open" if new words can continuously be added to the class, or relatively "closed" if there is a fixed number of words in a class. In English, the class of pronouns is closed, whereas the class of adjectives is open, since an infinite number of adjectives can be constructed from verbs (e.g. "saddened") or nouns (e.g. with the -like suffix, as in "noun-like"). In other languages such as Korean, the situation is the opposite, and new pronouns can be constructed, whereas the number of adjectives is fixed. Word classes also carry out differing functions in grammar. Prototypically, verbs are used to construct predicates, while nouns are used as arguments of predicates. In a sentence such as "Sally runs", the predicate is "runs", because it is the word that predicates a specific state about its argument "Sally". Some verbs such as "curse" can take two arguments, e.g. "Sally cursed John". A predicate that can only take a single argument is called intransitive, while a predicate that can take two arguments is called transitive. Many other word classes exist in different languages, such as conjunctions like "and" that serve to join two sentences, articles that introduce a noun, interjections such as "wow!", or ideophones like "splash" that mimic the sound of some event. Some languages have positionals that describe the spatial position of an event or entity. Many languages have classifiers that identify countable nouns as belonging to a particular type or having a particular shape. For instance, in Japanese, the general noun classifier for humans is nin (人), and it is used for counting humans: 

san-nin no gakusei (三人の学生) lit. "3 human-classifier of student" – three students For trees, it would be: 

san-bon no ki (三本の木) lit. "3 classifier-for-long-objects of tree" – three trees 

Morphology In linguistics, the study of the internal structure of complex words and the processes by which words are formed is called morphology. In most languages, it is possible to construct complex words that are built of several morphemes. For instance, the English word "unexpected" can be analyzed as being composed of the three morphemes "un-", "expect" and "-ed". Morphemes can be classified according to whether they are independent morphemes, so-called roots, or whether they can only co-occur attached to other morphemes. These bound morphemes or affixes can be classified according to their position in relation to the root: prefixes precede the root, suffixes follow the root, and infixes are inserted in the middle of a root. Affixes serve to modify or elaborate the meaning of the root. Some languages change the meaning of words by changing the phonological structure of a word, for example, the English word "run", which in the past tense is "ran". This process is called ablaut. Furthermore, morphology distinguishes between the process of inflection, which modifies or elaborates on a word, and the process of derivation, which creates a new word from an existing one. In English, the verb "sing" has the inflectional forms "singing" and "sung", which are both verbs, and the derivational form "singer", which is a noun derived from the verb with the agentive suffix "-er". Languages differ widely in how much they rely on morphological processes of word formation. In some languages, for example, Chinese, there are no morphological processes, and all grammatical information is encoded syntactically by forming strings of single words. This type of morpho-syntax is often called isolating, or analytic, because there is almost a full correspondence between a single word and a single aspect of meaning. Most languages have words consisting of several morphemes, but they vary in the degree to which morphemes are discrete units. In many languages, notably in most Indo-European languages, single morphemes may have several distinct meanings that cannot be analyzed into smaller segments. For example, in Latin, the word bonus, or "good", consists of the root bon-, meaning "good", and the suffix -us, which indicates masculine gender, singular number, and nominative case. These languages are called fusional languages, because several meanings may be fused into a single morpheme. The opposite of fusional languages are agglutinative languages which construct words by stringing morphemes together in chains, but with each morpheme as a discrete semantic unit. An example of such a language is Turkish, where for example, the word evlerinizden, or "from your houses", consists of the morphemes, ev-ler-iniz-den with the meanings house-plural-your-from. The languages that rely on morphology to the greatest extent are traditionally called polysynthetic languages. They may express the equivalent of an entire English sentence in a single word. For example, in Persian the single word نفهمیدمش, nafahmidamesh means I didn't understand it consisting of morphemes na-fahm-id-am-esh with the meanings, "negation.understand.past.I.it". As another example with more complexity, in the Yupik word tuntussuqatarniksatengqiggtuq, which means "He had not yet said again that he was going to hunt reindeer", the word consists of the morphemes tuntu-ssur-qatar-ni-ksaite-ngqiggte-uq with the meanings, "reindeer-hunt-future-say-negation-again-third.person.singular.indicative", and except for the morpheme tuntu ("reindeer") none of the other morphemes can appear in isolation. Many languages use morphology to cross-reference words within a sentence. This is sometimes called agreement. For example, in many Indo-European languages, adjectives must cross-reference the noun they modify in terms of number, case, and gender, so that the Latin adjective bonus, or "good", is inflected to agree with a noun that is masculine gender, singular number, and nominative case. In many polysynthetic languages, verbs cross-reference their subjects and objects. In these types of languages, a single verb may include information that would require an entire sentence in English. For example, in the Basque phrase ikusi nauzu, or "you saw me", the past tense auxiliary verb n-au-zu (similar to English "do") agrees with both the subject (you) expressed by the n- prefix, and with the object (me) expressed by the – zu suffix. The sentence could be directly transliterated as "see you-did-me" 

Syntax 

Another way in which languages convey meaning is through the order of words within a sentence. The grammatical rules for how to produce new sentences from words that are already known is called syntax. The syntactical rules of a language determine why a sentence in English such as "I love you" is meaningful, but "*love you I" is not. Syntactical rules determine how word order and sentence structure is constrained, and how those constraints contribute to meaning. For example, in English, the two sentences "the slaves were cursing the master" and "the master was cursing the slaves" mean different things, because the role of the grammatical subject is encoded by the noun being in front of the verb, and the role of object is encoded by the noun appearing after the verb. Conversely, in Latin, both Dominus servos vituperabat and Servos vituperabat dominus mean "the master was reprimanding the slaves", because servos, or "slaves", is in the accusative case, showing that they are the grammatical object of the sentence, and dominus, or "master", is in the nominative case, showing that he is the subject. Latin uses morphology to express the distinction between subject and object, whereas English uses word order. Another example of how syntactic rules contribute to meaning is the rule of inverse word order in questions, which exists in many languages. This rule explains why when in English, the phrase "John is talking to Lucy" is turned into a question, it becomes "Who is John talking to?", and not "John is talking to who?". The latter example may be used as a way of placing special emphasis on "who", thereby slightly altering the meaning of the question. Syntax also includes the rules for how complex sentences are structured by grouping words together in units, called phrases, that can occupy different places in a larger syntactic structure. Sentences can be described as consisting of phrases connected in a tree structure, connecting the phrases to each other at different levels. To the right is a graphic representation of the syntactic analysis of the English sentence "the cat sat on the mat". The sentence is analyzed as being constituted by a noun phrase, a verb, and a prepositional phrase; the prepositional phrase is further divided into a preposition and a noun phrase, and the noun phrases consist of an article and a noun. The reason sentences can be seen as being composed of phrases is because each phrase would be moved around as a single element if syntactic operations were carried out. For example, "the cat" is one phrase, and "on the mat" is another, because they would be treated as single units if a decision was made to emphasize the location by moving forward the prepositional phrase: "[And] on the mat, the cat sat". There are many different formalist and functionalist frameworks that propose theories for describing syntactic structures, based on different assumptions about what language is and how it should be described. Each of them would analyze a sentence such as this in a different manner. 

Typology and universals 

Languages can be classified in relation to their grammatical types. Languages that belong to different families nonetheless often have features in common, and these shared features tend to correlate. For example, languages can be classified on the basis of their basic word order, the relative order of the verb, and its constituents in a normal indicative sentence. In English, the basic order is SVO (subject–verb–object): "The snake(S) bit(V) the man(O)", whereas for example, the corresponding sentence in the Australian language Gamilaraay would be d̪uyugu n̪ama d̪ayn yiːy (snake man bit), SOV. Word order type is relevant as a typological parameter, because basic word order type corresponds with other syntactic parameters, such as the relative order of nouns and adjectives, or of the use of prepositions or postpositions. Such correlations are called implicational universals. For example, most (but not all) languages that are of the SOV type have postpositions rather than prepositions, and have adjectives before nouns. All languages structure sentences into Subject, Verb, and Object, but languages differ in the way they classify the relations between actors and actions. English uses the nominative-accusative word typology: in English transitive clauses, the subjects of both intransitive sentences ("I run") and transitive sentences ("I love you") are treated in the same way, shown here by the nominative pronoun I. Some languages, called ergative, Gamilaraay among them, distinguish instead between Agents and Patients. In ergative languages, the single participant in an intransitive sentence, such as "I run", is treated the same as the patient in a transitive sentence, giving the equivalent of "me run". Only in transitive sentences would the equivalent of the pronoun "I" be used. In this way the semantic roles can map onto the grammatical relations in different ways, grouping an intransitive subject either with Agents (accusative type) or Patients (ergative type) or even making each of the three roles differently, which is called the tripartite type. The shared features of languages which belong to the same typological class type may have arisen completely independently. Their co-occurrence might be due to universal laws governing the structure of natural languages, "language universals", or they might be the result of languages evolving convergent solutions to the recurring communicative problems that humans use language to solve. 

Social contexts of use and transmission 

While humans have the ability to learn any language, they only do so if they grow up in an environment in which language exists and is used by others. Language is therefore dependent on communities of speakers in which children learn language from their elders and peers and themselves transmit language to their own children. Languages are used by those who speak them to communicate and to solve a plethora of social tasks. Many aspects of language use can be seen to be adapted specifically to these purposes. Owing to the way in which language is transmitted between generations and within communities, language perpetually changes, diversifying into new languages or converging due to language contact. The process is similar to the process of evolution, where the process of descent with modification leads to the formation of a phylogenetic tree. However, languages differ from biological organisms in that they readily incorporate elements from other languages through the process of diffusion, as speakers of different languages come into contact. Humans also frequently speak more than one language, acquiring their first language or languages as children, or learning new languages as they grow up. Because of the increased language contact in the globalizing world, many small languages are becoming endangered as their speakers shift to other languages that afford the possibility to participate in larger and more influential speech communities. 

Usage and meaning 

When studying the way in which words and signs are used, it is often the case that words have different meanings, depending on the social context of use. An important example of this is the process called deixis, which describes the way in which certain words refer to entities through their relation between a specific point in time and space when the word is uttered. Such words are, for example, the word, "I" (which designates the person speaking), "now" (which designates the moment of speaking), and "here" (which designates the position of speaking). Signs also change their meanings over time, as the conventions governing their usage gradually change. The study of how the meaning of linguistic expressions changes depending on context is called pragmatics. Deixis is an important part of the way that we use language to point out entities in the world. Pragmatics is concerned with the ways in which language use is patterned and how these patterns contribute to meaning. For example, in all languages, linguistic expressions can be used not just to transmit information, but to perform actions. Certain actions are made only through language, but nonetheless have tangible effects, e.g. the act of "naming", which creates a new name for some entity, or the act of "pronouncing someone man and wife", which creates a social contract of marriage. These types of acts are called speech acts, although they can also be carried out through writing or hand signing. The form of linguistic expression often does not correspond to the meaning that it actually has in a social context. For example, if at a dinner table a person asks, "Can you reach the salt?", that is, in fact, not a question about the length of the arms of the one being addressed, but a request to pass the salt across the table. This meaning is implied by the context in which it is spoken; these kinds of effects of meaning are called conversational implicatures. These social rules for which ways of using language are considered appropriate in certain situations and how utterances are to be understood in relation to their context vary between communities, and learning them is a large part of acquiring communicative competence in a language. 

Acquisition 

All healthy, normally developing human beings learn to use language. Children acquire the language or languages used around them: whichever languages they receive sufficient exposure to during childhood. The development is essentially the same for children acquiring sign or oral languages. This learning process is referred to as first-language acquisition, since unlike many other kinds of learning, it requires no direct teaching or specialized study. In The Descent of Man, naturalist Charles Darwin called this process "an instinctive tendency to acquire an art". 

First language acquisition proceeds in a fairly regular sequence, though there is a wide degree of variation in the timing of particular stages among normally developing infants. Studies published in 2013 have indicated that unborn fetuses are capable of language acquisition to some degree. From birth, newborns respond more readily to human speech than to other sounds. Around one month of age, babies appear to be able to distinguish between different speech sounds. Around six months of age, a child will begin babbling, producing the speech sounds or handshapes of the languages used around them. Words appear around the age of 12 to 18 months; the average vocabulary of an eighteen-month-old child is around 50 words. A child's first utterances are holophrases (literally "whole-sentences"), utterances that use just one word to communicate some idea. Several months after a child begins producing words, the child will produce two-word utterances, and within a few more months will begin to produce telegraphic speech, or short sentences that are less grammatically complex than adult speech, but that do show regular syntactic structure. From roughly the age of three to five years, a child's ability to speak or sign is refined to the point that it resembles adult language. Acquisition of second and additional languages can come at any age, through exposure in daily life or courses. Children learning a second language are more likely to achieve native-like fluency than adults, but in general, it is very rare for someone speaking a second language to pass completely for a native speaker. An important difference between first language acquisition and additional language acquisition is that the process of additional language acquisition is influenced by languages that the learner already knows. 

Culture 

Languages, understood as the particular set of speech norms of a particular community, are also a part of the larger culture of the community that speaks them. Languages differ not only in pronunciation, vocabulary, and grammar, but also through having different "cultures of speaking." Humans use language as a way of signalling identity with one cultural group as well as difference from others. Even among speakers of one language, several different ways of using the language exist, and each is used to signal affiliation with particular subgroups within a larger culture. Linguists and anthropologists, particularly sociolinguists, ethnolinguists, and linguistic anthropologists have specialized in studying how ways of speaking vary between speech communities. Linguists use the term "varieties" to refer to the different ways of speaking a language. This term includes geographically or socioculturally defined dialects as well as the jargons or styles of subcultures. Linguistic anthropologists and sociologists of language define communicative style as the ways that language is used and understood within a particular culture. Because norms for language use are shared by members of a specific group, communicative style also becomes a way of displaying and constructing group identity. Linguistic differences may become salient markers of divisions between social groups, for example, speaking a language with a particular accent may imply membership of an ethnic minority or social class, one's area of origin, or status as a second language speaker. These kinds of differences are not part of the linguistic system, but are an important part of how people use language as a social tool for constructing groups. However, many languages also have grammatical conventions that signal the social position of the speaker in relation to others through the use of registers that are related to social hierarchies or divisions. In many languages, there are stylistic or even grammatical differences between the ways men and women speak, between age groups, or between social classes, just as some languages employ different words depending on who is listening. For example, in the Australian language Dyirbal, a married man must use a special set of words to refer to everyday items when speaking in the presence of his mother-in-law. Some cultures, for example, have elaborate systems of "social deixis", or systems of signalling social distance through linguistic means. In English, social deixis is shown mostly through distinguishing between addressing some people by first name and others by surname, and in titles such as "Mrs.", "boy", "Doctor", or "Your Honor", but in other languages, such systems may be highly complex and codified in the entire grammar and vocabulary of the language. For instance, in languages of east Asia such as Thai, Burmese, and Javanese, different words are used according to whether a speaker is addressing someone of higher or lower rank than oneself in a ranking system with animals and children ranking the lowest and gods and members of royalty as the highest. 

Writing, literacy and technology 

Throughout history a number of different ways of representing language in graphic media have been invented. These are called writing systems. The use of writing has made language even more useful to humans. It makes it possible to store large amounts of information outside of the human body and retrieve it again, and it allows communication across physical distances and timespans that would otherwise be impossible. Many languages conventionally employ different genres, styles, and registers in written and spoken language, and in some communities, writing traditionally takes place in an entirely different language than the one spoken. There is some evidence that the use of writing also has effects on the cognitive development of humans, perhaps because acquiring literacy generally requires explicit and formal education. The invention of the first writing systems is roughly contemporary with the beginning of the Bronze Age in the late 4th millennium BC. The Sumerian archaic cuneiform script and the Egyptian hieroglyphs are generally considered to be the earliest writing systems, both emerging out of their ancestral proto-literate symbol systems from 3400 to 3200 BC with the earliest coherent texts from about 2600 BC. It is generally agreed that Sumerian writing was an independent invention; however, it is debated whether Egyptian writing was developed completely independently of Sumerian, or was a case of cultural diffusion. A similar debate exists for the Chinese script, which developed around 1200 BC. The pre-Columbian Mesoamerican writing systems (including among others Olmec and Maya scripts) are generally believed to have had independent origins. 

Change 

All languages change as speakers adopt or invent new ways of speaking and pass them on to other members of their speech community. Language change happens at all levels from the phonological level to the levels of vocabulary, morphology, syntax, and discourse. Even though language change is often initially evaluated negatively by speakers of the language who often consider changes to be "decay" or a sign of slipping norms of language usage, it is natural and inevitable. Changes may affect specific sounds or the entire phonological system. Sound change can consist of the replacement of one speech sound or phonetic feature by another, the complete loss of the affected sound, or even the introduction of a new sound in a place where there had been none. Sound changes can be conditioned in which case a sound is changed only if it occurs in the vicinity of certain other sounds. Sound change is usually assumed to be regular, which means that it is expected to apply mechanically whenever its structural conditions are met, irrespective of any non-phonological factors. On the other hand, sound changes can sometimes be sporadic, affecting only one particular word or a few words, without any seeming regularity. Sometimes a simple change triggers a chain shift in which the entire phonological system is affected. This happened in the Germanic languages when the sound change known as Grimm's law affected all the stop consonants in the system. The original consonant *bʰ became /b/ in the Germanic languages, the previous *b in turn became /p/, and the previous *p became /f/. The same process applied to all stop consonants and explains why Italic languages such as Latin have p in words like pater and pisces, whereas Germanic languages, like English, have father and fish. Another example is the Great Vowel Shift in English, which is the reason that the spelling of English vowels do not correspond well to their current pronunciation. This is because the vowel shift brought the already established orthography out of synchronization with pronunciation. Another source of sound change is the erosion of words as pronunciation gradually becomes increasingly indistinct and shortens words, leaving out syllables or sounds. This kind of change caused Latin mea domina to eventually become the French madame and American English ma'am. Change also happens in the grammar of languages as discourse patterns such as idioms or particular constructions become grammaticalized. This frequently happens when words or morphemes erode and the grammatical system is unconsciously rearranged to compensate for the lost element. For example, in some varieties of Caribbean Spanish the final /s/ has eroded away. Since Standard Spanish uses final /s/ in the morpheme marking the second person subject "you" in verbs, the Caribbean varieties now have to express the second person using the pronoun tú. This means that the sentence "what's your name" is ¿como te llamas? [ˈkomo te ˈjamas] in Standard Spanish, but [ˈkomo ˈtu te ˈjama] in Caribbean Spanish. The simple sound change has affected both morphology and syntax. Another common cause of grammatical change is the gradual petrification of idioms into new grammatical forms, for example, the way the English "going to" construction lost its aspect of movement and in some varieties of English has almost become a full-fledged future tense (e.g. I'm gonna). Language change may be motivated by "language internal" factors, such as changes in pronunciation motivated by certain sounds being difficult to distinguish aurally or to produce, or through patterns of change that cause some rare types of constructions to drift towards more common types. Other causes of language change are social, such as when certain pronunciations become emblematic of membership in certain groups, such as social classes, or with ideologies, and therefore are adopted by those who wish to identify with those groups or ideas. In this way, issues of identity and politics can have profound effects on language structure. 

Contact 

One source of language change is contact and the resulting diffusion of linguistic traits between languages. Language contact occurs when speakers of two or more languages or varieties interact on a regular basis. Multilingualism is likely to have been the norm throughout human history and most people in the modern world are multilingual. Before the rise of the concept of the ethno-national state, monolingualism was characteristic mainly of populations inhabiting small islands. But with the ideology that made one people, one state, and one language the most desirable political arrangement, monolingualism started to spread throughout the world. There are only 250 countries in the world corresponding to some 6,000 languages, which means that most countries are multilingual and most languages therefore exist in close contact with other languages. When speakers of different languages interact closely, it is typical for their languages to influence each other. Through sustained language contact over long periods, linguistic traits diffuse between languages, and languages belonging to different families may converge to become more similar. In areas where many languages are in close contact, this may lead to the formation of language areas in which unrelated languages share a number of linguistic features. A number of such language areas have been documented, among them, the Balkan language area, the Mesoamerican language area, and the Ethiopian language area. Also, larger areas such as South Asia, Europe, and Southeast Asia have sometimes been considered language areas because of the widespread diffusion of specific areal features. 

Language contact may also lead to a variety of other linguistic phenomena, including language convergence, borrowing, and relexification (the replacement of much of the native vocabulary with that of another language). In situations of extreme and sustained language contact, it may lead to the formation of new mixed languages that cannot be considered to belong to a single language family. One type of mixed language called pidgins occurs when adult speakers of two different languages interact on a regular basis, but in a situation where neither group learns to speak the language of the other group fluently. In such a case, they will often construct a communication form that has traits of both languages, and that has a simplified grammatical and phonological structure. The language comes to contain mostly the grammatical and phonological categories that exist in both languages. Pidgin languages are defined by not having any native speakers, but only being spoken by people who have another language as their first language. But if the Pidgin language becomes the main language of a speech community, then eventually children will grow up learning the Pidgin language as their first language. As the generation of child learners grows up, the pidgin will often be seen to change its structure and acquire a greater degree of complexity. This type of language is generally called a creole language. An example of such mixed languages is Tok Pisin, the official language of Papua New Guinea, which originally arose as a Pidgin based on English and Austronesian languages; others are Kreyòl ayisyen, the French-based creole language spoken in Haiti, and Michif, a mixed language of Canada, based on the Native American language Cree and French. 

Linguistic diversity 

SIL Ethnologue defines a "living language" as "one that has at least one speaker for whom it is their first language". The exact number of known living languages varies from 6,000 to 7,000, depending on the precision of one's definition of "language", and in particular, on how one defines the distinction between a "language" and a "dialect". As of 2026, Ethnologue cataloged 7,170 living human languages. The Ethnologue establishes linguistic groups based on studies of mutual intelligibility, and therefore often includes more categories than more conservative classifications. For example, the Danish language that most scholars consider a single language with several dialects is classified as two distinct languages (Danish and Jutish) by the Ethnologue. According to the Ethnologue, 389 languages (nearly 6%) have more than a million speakers. These languages together account for 94% of the world's population, whereas 94% of the world's languages account for the remaining 6% of the global population. 

Languages and dialects 

There is no clear distinction between a language and a dialect, notwithstanding a famous aphorism attributed to linguist Max Weinreich that "a language is a dialect with an army and navy". For example, national boundaries frequently override linguistic difference in determining whether two linguistic varieties are languages or dialects. Hakka, Cantonese and Mandarin are, for example, often classified as "dialects" of Chinese, even though they are more different from each other than Swedish is from Norwegian. Before the Yugoslav Wars, Serbo-Croatian was generally considered a single language with two normative variants, but due to sociopolitical reasons, Croatian and Serbian are now often treated as separate languages and employ different writing systems. In other words, the distinction may hinge on political considerations as much as on cultural differences as on distinctive writing systems or the degree of mutual intelligibility. The latter is, in fact, a rather unreliable criterion to discriminate languages and dialects. Pluricentric languages, which are languages with more than one standard variety, are a case in point. Standard American English and Standard RP (English) English, for instance, may in some areas be more different than languages with names, e.g. Swedish and Norwegian. A complex social process of "language making" underlies these assignments of status and in some cases even linguistic experts may not agree (e.g. the One Standard German Axiom). The language making process is dynamic and subject to change over time. 

Language families of the world 

The world's languages can be grouped into language families consisting of languages that can be shown to have common ancestry. Linguists recognize many hundreds of language families, although some of them can possibly be grouped into larger units as more evidence becomes available and in-depth studies are carried out. At present, there are also dozens of language isolates: languages that cannot be shown to be related to any other languages in the world. Among them are Basque, spoken in Europe, Zuni of New Mexico, Purépecha of Mexico, Ainu of Japan, Burushaski of Pakistan, and many others. The language family of the world that has the most speakers is the Indo-European languages, spoken by 46% of the world's population. This family includes major world languages like English, Spanish, French, German, Russian, and Hindustani (Hindi/Urdu). The Indo-European family spread first through hypothesized Indo-European migrations that would have taken place some time in the period c. 8000–1500 BCE, and subsequently through much later European colonial expansion, which brought the Indo-European languages to a politically and often numerically dominant position in the Americas and much of Africa. The Sino-Tibetan languages are spoken by 20% of the world's population and include many of the languages of East Asia, including Hakka, Mandarin Chinese, Cantonese, and hundreds of smaller languages. Africa is home to a large number of language families, the largest of which is the Niger-Congo language family, which includes such languages as Swahili, Shona, and Yoruba. Speakers of the Niger-Congo languages account for 6.9% of the world's population. A similar number of people speak the Afroasiatic languages, which include the populous Semitic languages such as Arabic, Hebrew language, and the languages of the Sahara region, such as the Berber languages and Hausa. The Austronesian languages are spoken by 5.5% of the world's population and stretch from Madagascar to maritime Southeast Asia all the way to Oceania. It includes such languages as Malagasy, Māori, Samoan, and many of the indigenous languages of Indonesia and Taiwan. The Austronesian languages are considered to have originated in Taiwan around 3000 BC and spread through the Oceanic region through island-hopping, based on an advanced nautical technology. Other populous language families are the Dravidian languages of South Asia (among them Kannada, Tamil, and Telugu), the Turkic languages of Central Asia (such as Turkish), the Austroasiatic (among them Khmer), and Tai–Kadai languages of Southeast Asia (including Thai). The areas of the world in which there is the greatest linguistic diversity, such as the Americas, Papua New Guinea, West Africa, and South-Asia, contain hundreds of small language families. These areas together account for the majority of the world's languages, though not the majority of speakers. In the Americas, some of the largest language families include the Quechua, Arawak, and Tupi-Guarani families of South America, the Uto-Aztecan, Oto-Manguean, and Mayan of Mesoamerica, and the Na-Dene, Iroquoian, and Algonquian language families of North America. In Australia, most indigenous languages belong to the Pama-Nyungan family, whereas New Guinea is home to a large number of small families and isolates, as well as a number of Austronesian languages. Due to its remoteness and geographical fragmentation, Papua New Guinea emerges in fact as the leading location worldwide for both species (8% of world total) and linguistic richness – with 830 living tongues (12% of world total). 

Language endangerment 

Language endangerment occurs when a language is at risk of falling out of use as its speakers die out or shift to speaking another language. Language loss occurs when the language has no more native speakers, and becomes a dead language. If eventually no one speaks the language at all, it becomes an extinct language. While languages have always gone extinct throughout human history, they have been disappearing at an accelerated rate in the 20th and 21st centuries due to the processes of globalization and neo-colonialism, where the economically powerful languages dominate other languages. The more commonly spoken languages dominate the less commonly spoken languages, so the less commonly spoken languages eventually disappear from populations. Of the between 6,000 and 7,000 languages spoken as of 2010, between 50 and 90% of those are expected to have become extinct by the year 2100. The top 20 languages, those spoken by more than 50 million speakers each, are spoken by 50% of the world's population, whereas many of the other languages are spoken by smaller communities, most of them with less than 10,000 speakers. 

The United Nations Educational, Scientific and Cultural Organization (UNESCO) operates with five levels of language endangerment: "safe", "vulnerable" (not spoken by children outside the home), "definitely endangered" (not spoken by children), "severely endangered" (only spoken by the oldest generations), and "critically endangered" (spoken by a few members of the oldest generation, often semi-speakers). Despite claims that the world would be better off if most adopted a single common lingua franca, such as English or Esperanto, there is a consensus that the loss of languages harms the cultural diversity of the world. It is a common belief, going back to the Tower of Babel narrative in the Hebrew Bible, that linguistic diversity causes political conflict, but many of the world's major episodes of violence have taken place in situations with low linguistic diversity, such as the Yugoslav and American Civil War, or the genocide of Rwanda. Many projects aim to prevent or slow this loss by revitalizing endangered languages and promoting education and literacy in minority languages. Across the world, many countries have enacted specific legislation to protect and stabilize the language of indigenous speech communities. A minority of linguists have argued that language loss is a natural process that should not be counteracted and that documenting endangered languages for posterity is sufficient. The University of Waikato is using the Welsh language as a model for their Māori language revitalisation programme, as they deem Welsh to be the world's leading example for the survival of languages. In 2019, Hawaiian TV company Oiwi visited a Welsh language centre in Nant Gwrtheyrn, North Wales, to help find ways of preserving their Ōlelo Hawaiʻi language. 

See also 

Notes 

References 

Works cited 

Further reading Crystal, David (1997). The Cambridge Encyclopedia of Language. Cambridge: Cambridge University Press. Cysouw, Michael; Good, Jeff (2013). "Languoid, doculect and glossonym: Formalizing the notion 'language'". Language Documentation and Conservation. 7: 331–359. hdl:10125/4606. Allison Parshall, "Pain Language: The sound of 'ow' transcends borders", Scientific American, vol. 332, no. 2 (February 2025), pp. 16–18. "Many languages have an interjection word for expressing pain. [Katarzyna Pisanski et al., writing in the Journal of the Acoustical Society of America, have] found that pain interjections tend to contain the vowel sound 'ah' (written as [a] in the International Phonetic Alphabet) and letter combinations that incorporate it, such as 'ow' and 'ai.' These patterns may point back to the origins of human language itself." (p. 16.) "Researchers are continually discovering cases of symbolism, or sound iconicity, in which a word's intrinsic nature has some connection to its meaning. These cases run counter to decades of linguistic theory, which had regarded language as fundamentally arbitrary... [Many words onomatopoeically imitate a sound. Also] there's the 'bouba-kiki' effect, whereby people from varying cultures are more likely to associate the nonsense word 'bouba' with a rounded shape and 'kiki' with a spiked one... [S]omehow we all have a feeling about this,' says Aleksandra Ćwiek... [She and her colleagues have] show[n] that people associate the trilled 'R' sound with roughness and the 'L' sound with smoothness. Mark Dingemanse... in 2013 found [that] the conversational 'Huh?' and similar words in other languages may be universal." (p. 18.) Stix, Gary, "Thinking without Words: Cognition doesn't require language, it turns out" (interview with Evelina Fedorenko, a cognitive neuroscientist at the Massachusetts Institute of Technology), Scientific American, vol. 332, no. 3 (March 2025), pp. 86–88. "[I]n the tradition of linguist Noam Chomsky... we use language for thinking: to think is why language evolved in our species. [However, evidence that thought and language are separate systems is found, for example, by] looking at deficits in different abilities – for instance, in people with brain damage... who have impairments in language – some form of aphasia [ – yet are clearly able to think]." (p. 87.) Conversely, "large language models such as GPT-2... do language very well [but t]hey're not so good at thinking, which... nicely align[s] with the idea that the language system by itself is not what makes you think." (p. 88.) Swadesh, Morris (1934). "The phonemic principle". Language. 10 (2): 117–129. doi:10.2307/409603. JSTOR 409603. 

External links 

World Atlas of Language Structures: a large database of structural (phonological, grammatical, lexical) properties of languages Ethnologue: Languages of the World is a comprehensive catalog of all of the world's known living languages 

[Civilization] A civilization (American English and Oxford spelling) or civilisation (common British English) is any complex society characterized by the development of the state, urbanization, and standardized symbolic systems of communication, namely writing systems.Civilizations are organized around densely populated settlements, divided into more or less rigid hierarchical social classes of division of labour, often with a ruling elite and subordinate urban and rural populations, which engage in intensive agriculture, mining, small-scale manufacture and trade. Civilization concentrates power, extending human control over the rest of nature, including over other human beings. Civilizations are characterized by elaborate agriculture, architecture, infrastructure, technological advancement, currency, taxation, regulation, and specialization of labour. 

Historically, a civilization has often been understood as a larger and "more advanced" culture, in implied contrast to smaller, supposedly less advanced cultures, even societies within civilizations themselves and within their histories. Generally civilization contrasts with non-centralized tribal societies, including the cultures of nomadic pastoralists, Neolithic societies, or hunter-gatherers. The word civilization relates to the Latin civitas or 'city'. As the National Geographic Society has explained it: "This is why the most basic definition of the word civilization is 'a society made up of cities.'" The earliest emergence of civilizations is generally connected with the final stages of the Neolithic Revolution in West Asia, culminating in the relatively rapid process of urban revolution and state formation, a political development associated with the appearance of a governing elite. 

History of the concept The English word civilization comes from the French civilisé ('civilized'), from Latin: civilis ('civil'), related to civis ('citizen') and civitas ('city'). The fundamental treatise is Norbert Elias's The Civilizing Process (1939), which traces social mores from medieval courtly society to the early modern period. In The Philosophy of Civilization (1923), Albert Schweitzer outlines two opinions: one purely material and the other material and ethical. He said that the world crisis was from humanity losing the ethical idea of civilization, "the sum total of all progress made by man in every sphere of action and from every point of view in so far as the progress helps towards the spiritual perfecting of individuals as the progress of all progress". Related words like "civility" developed in the mid-16th century. The abstract noun "civilization", meaning "civilized condition", came in the 1760s, again from French. The first known use in French is in 1757, by Victor de Riqueti, marquis de Mirabeau, and the first use in English is attributed to Adam Ferguson, who in his 1767 Essay on the History of Civil Society wrote, "Not only the individual advances from infancy to manhood but the species itself from rudeness to civilisation". The word was therefore opposed to barbarism or rudeness, in the active pursuit of progress characteristic of the Age of Enlightenment. In the late 1700s and early 1800s, during the French Revolution, "civilization" was used in the singular, never in the plural, and meant the progress of humanity as a whole. This is still the case in French. The use of "civilizations" as a countable noun was in occasional use in the 19th century, but has become much more common in the later 20th century, sometimes just meaning culture (itself in origin an uncountable noun, made countable in the context of ethnography). Only in this generalized sense does it become possible to speak of a "medieval civilization", which in Elias's sense would have been an oxymoron. Using the terms "civilization" and "culture" as equivalents is controversial and generally rejected, so that, for example, some types of culture are not normally described as civilizations. Already in the 18th century, civilization was not always seen as an improvement. One historically important distinction between culture and civilization is from the writings of Rousseau, particularly his work about education, Emile. Here, civilization, being more rational and socially driven, is not fully in accord with human nature, and "human wholeness is achievable only through the recovery of or approximation to an original discursive or pre-rational natural unity" (see noble savage). From this, a new approach was developed, especially in Germany, first by Johann Gottfried Herder and later by philosophers such as Kierkegaard and Nietzsche. This sees cultures as natural organisms, not defined by "conscious, rational, deliberative acts", but a kind of pre-rational "folk spirit". Civilization, in contrast, though more rational and more successful in material progress, is unnatural and leads to "vices of social life" such as guile, hypocrisy, envy and avarice. In World War II, Leo Strauss, having fled Germany, argued in New York that this opinion of civilization was behind Nazism and German militarism and nihilism. 

Characteristics 

Social scientists such as V. Gordon Childe have named a number of traits that distinguish a civilization from other kinds of society. Civilizations have been distinguished by their means of subsistence, types of livelihood, settlement patterns, forms of government, social stratification, economic systems, literacy and other cultural traits. Andrew Nikiforuk argues that "civilizations relied on shackled human muscle. It took the energy of slaves to plant crops, clothe emperors, and build cities" and considers slavery to be a common feature of pre-modern civilizations. All civilizations have depended on agriculture for subsistence, with the possible exception of some early civilizations in Peru which may have depended upon maritime resources. Most developed and permanent civilizations depended on cereal agriculture. The traditional "surplus model" postulates that cereal farming results in accumulated storage and a surplus of food, particularly when people use intensive agricultural techniques such as artificial fertilization, irrigation and crop rotation. It is possible but more difficult to accumulate horticultural production, and so civilizations based on horticultural gardening have been very rare. Grain surpluses have been especially important because grain can be stored for a long time. Research from the Journal of Political Economy contradicts the surplus model. It postulates that horticultural gardening was more productive than cereal farming. However, only cereal farming produced civilization because of the appropriability of the yearly harvest. Rural populations that could only grow cereals could be taxed allowing for a taxing elite and urban development. This also had a negative effect on the rural population, increasing relative agricultural output per farmer. Farming efficiency created food surplus and sustained the food surplus through decreasing rural population growth in favour of urban growth. Suitability of highly productive roots and tubers was in fact a curse of plenty, which prevented the emergence of states and impeded economic development. A surplus of food permits some people to do things besides producing food for a living: early civilizations included soldiers, artisans, priests and priestesses, and other people with specialized careers. A surplus of food results in a division of labour and a more diverse range of human activity, a defining trait of civilizations. However, in some places hunter-gatherers have had access to food surpluses, such as among some of the indigenous peoples of the Pacific Northwest and perhaps during the Mesolithic Natufian culture. It is possible that food surpluses and relatively large scale social organization and division of labour predates plant and animal domestication. Civilizations have distinctly different settlement patterns from other societies. The word civilization is sometimes defined as "living in cities". Non-farmers tend to gather in cities to work and to trade. Compared with other societies, civilizations have a more complex political structure, namely the state. State societies are more stratified than other societies; there is a greater difference among the social classes. The ruling class, normally concentrated in the cities, has control over much of the surplus and exercises its will through the actions of a government or bureaucracy. Morton Fried, a conflict theorist and Elman Service, an integration theorist, have classified human cultures based on political systems and social inequality. This system of classification contains four categories. 

Hunter-gatherer bands, which are generally egalitarian. Horticultural–pastoralist societies in which there are generally two inherited social classes: chief and commoner. Highly stratified structures, or chiefdoms, with several inherited social classes: king, noble, freemen, serf and slave. Civilizations, with complex social hierarchies and organized, institutional forms of government. Civilizations can show varying degrees of social mobility and social stratification. Economically, civilizations display more complex patterns of ownership and exchange than less organized societies. Living in one place allows people to accumulate more personal possessions than nomadic people. Some people also acquire landed property, or private ownership of the land. Because a percentage of people in civilizations do not grow their own food, they must trade their goods and services for food in a market system, or receive food through the levy of tribute, redistributive taxation, tariffs or tithes from the food producing segment of the population. Early human cultures functioned through a gift economy supplemented by limited barter systems. By the early Iron Age, contemporary civilizations developed money as a medium of exchange for increasingly complex transactions. In a village, the potter makes a pot for the brewer and the brewer compensates the potter by giving him a certain amount of beer. In a city, the potter may need a new roof, the roofer may need new shoes, the cobbler may need new horseshoes, the blacksmith may need a new coat and the tanner may need a new pot. These people may not be personally acquainted with one another and their needs may not occur all at the same time. A monetary system is a way of organizing these obligations to ensure that they are fulfilled. From the days of the earliest monetarized civilizations, monopolistic controls of monetary systems have benefited the social and political elites. The transition from simpler to more complex economies does not necessarily mean an improvement in the living standards of the populace. For example, although the Middle Ages is often portrayed as an era of decline from the Roman Empire, studies have shown that the average stature of males in the Middle Ages (c. 500 to 1500 CE) was greater than it was for males during the preceding Roman Empire and the succeeding Early Modern Period (c. 1500 to 1800 CE). Also, the Plains Indians of North America in the 19th century were taller than their "civilized" American and European counterparts. The average stature of a population is a good measurement of the adequacy of its access to necessities, especially food, and its freedom from disease. Writing, developed first by people in Sumer, is considered a hallmark of civilization and "appears to accompany the rise of complex administrative bureaucracies or the conquest state". Traders and bureaucrats relied on writing to keep accurate records. Like money, the writing was necessitated by the size of the population of a city and the complexity of its commerce among people who are not all personally acquainted with each other. However, writing is not always necessary for civilization, as shown by the Inca civilization of the Andes, which did not use writing at all but except for a complex recording system consisting of knotted strings of different lengths and colours: the quipus, and still functioned as a civilized society. Aided by their division of labour and central government planning, civilizations have developed many other diverse cultural traits. These include organized religion, development in the arts, and countless new advances in science and technology. Assessments of what level of civilization a polity has reached are based on comparisons of the relative importance of agricultural as opposed to trading or manufacturing capacities, the territorial extensions of its power, the complexity of its division of labour, and the carrying capacity of its urban centres. Secondary elements include a developed transportation system, writing, standardized measurement, currency, contractual and tort-based legal systems, art, architecture, mathematics, scientific understanding, metallurgy, political structures, and organized religion. 

As a contrast with other societies The idea of civilization implies a progression or development from a previous "uncivilized" state. Traditionally, cultures that defined themselves as "civilized" often did so in contrast to other societies or human groupings viewed as less civilized, calling the latter barbarians, savages, and primitives. Indeed, the modern Western idea of civilization developed as a contrast to the indigenous cultures European settlers encountered during the European colonization of the Americas and Australia. The term "primitive," though once used in anthropology, has now been largely condemned by anthropologists because of its derogatory connotations and because it implies that the cultures it refers to are relics of a past time that do not change or progress. Because of this, societies regarding themselves as "civilized" have sometimes sought to dominate and assimilate "uncivilized" cultures into a "civilized" way of living. In the 19th century, the idea of European culture as "civilized" and superior to "uncivilized" non-European cultures was fully developed, and civilization became a core part of European identity. The idea of civilization can also be used as a justification for dominating another culture and dispossessing a people of their land. For example, in Australia, British settlers justified the displacement of Indigenous Australians by observing that the land appeared uncultivated and wild, which to them reflected that the inhabitants were not civilized enough to "improve" it. The behaviours and modes of subsistence that characterize civilization have been spread by colonization, invasion, religious conversion, the extension of bureaucratic control and trade, and by the introduction of new technologies to cultures that did not previously have them. Though aspects of culture associated with civilization can be freely adopted through contact between cultures, since early modern times Eurocentric ideals of "civilization" have been widely imposed upon cultures through coercion and dominance. These ideals complemented a philosophy that assumed there were innate differences between "civilized" and "uncivilized" peoples. 

Cultural identity 

"Civilization" can also refer to the culture of a complex society, not just the society itself. Every society, civilization or not, has a specific set of ideas and customs, and a certain set of manufactures and arts that make it unique. Civilizations tend to develop intricate cultures, including a state-based decision-making apparatus, a literature, professional art, architecture, organized religion and complex customs of education, coercion and control associated with maintaining the elite. The intricate culture associated with civilization has a tendency to spread to and influence other cultures, sometimes assimilating them into the civilization, a classic example being Chinese civilization and its influence on nearby civilizations such as Korea, Japan and Vietnam Many civilizations are actually large cultural spheres containing many nations and regions. The civilization in which someone lives is that person's broadest cultural identity. 

It is precisely the protection of this cultural identity that is becoming increasingly important nationally and internationally. According to international law, the United Nations and UNESCO try to set up and enforce relevant rules. The aim is to preserve the cultural heritage of humanity and also the cultural identity, especially in the case of war and armed conflict. According to Karl von Habsburg, President of Blue Shield International, the destruction of cultural assets is also part of psychological warfare. The target of the attack is often the opponent's cultural identity, which is why symbolic cultural assets become a main target. It is also intended to destroy the particularly sensitive cultural memory (museums, archives, monuments, etc.), the grown cultural diversity, and the economic basis (such as tourism) of a state, region or community. Many historians have focused on these broad cultural spheres and have treated civilizations as discrete units. Early twentieth-century philosopher Oswald Spengler, uses the German word Kultur, "culture", for what many call a "civilization". Spengler believed a civilization's coherence is based on a single primary cultural symbol. Cultures experience cycles of birth, life, decline, and death, often supplanted by a potent new culture, formed around a compelling new cultural symbol. Spengler states civilization is the beginning of the decline of a culture as "the most external and artificial states of which a species of developed humanity is capable". This "unified culture" concept of civilization also influenced the theories of historian Arnold J. Toynbee in the mid-twentieth century. Toynbee explored civilization processes in his multi-volume A Study of History, which traced the rise and, in most cases, the decline of 21 civilizations and five "arrested civilizations". Civilizations generally declined and fell, according to Toynbee, because of the failure of a "creative minority", through moral or religious decline, to meet some important challenge, rather than mere economic or environmental causes. Samuel P. Huntington defines civilization as "the highest cultural grouping of people and the broadest level of cultural identity people have short of that which distinguishes humans from other species". 

Complex systems 

Another group of theorists, making use of systems theory, looks at a civilization as a complex system, i.e., a framework by which a group of objects can be analysed that work in concert to produce some result. Civilizations can be seen as networks of cities that emerge from pre-urban cultures and are defined by the economic, political, military, diplomatic, social and cultural interactions among them. Any organization is a complex social system and a civilization is a large organization. Systems theory helps guard against superficial and misleading analogies in the study and description of civilizations. Systems theorists look at many types of relations between cities, including economic relations, cultural exchanges and political/diplomatic/military relations. These spheres often occur on different scales. For example, trade networks were, until the nineteenth century, much larger than either cultural spheres or political spheres. Extensive trade routes, including the Silk Road through Central Asia and Indian Ocean sea routes linking the Roman Empire, Persian Empire, India and China, were well established 2000 years ago when these civilizations scarcely shared any political, diplomatic, military, or cultural relations. The first evidence of such long-distance trade is in the ancient world. During the Uruk period, Guillermo Algaze has argued that trade relations connected Egypt, Mesopotamia, Iran and Afghanistan. Resin found later in the Royal Cemetery at Ur is suggested was traded northwards from Mozambique. Many theorists argue that the entire world has already become integrated into a single "world system", a process known as globalization. Different civilizations and societies all over the globe are economically, politically, and even culturally interdependent in many ways. There is debate over when this integration began, and what sort of integration – cultural, technological, economic, political, or military-diplomatic – is the key indicator in determining the extent of a civilization. David Wilkinson has proposed that economic and military-diplomatic integration of the Mesopotamian and Egyptian civilizations resulted in the creation of what he calls the "Central Civilization" around 1500 BCE. Central Civilization later expanded to include the entire Middle East and Europe, and then expanded to a global scale with European colonization, integrating the Americas, Australia, China and Japan by the nineteenth century. According to Wilkinson, civilizations can be culturally heterogeneous, like the Central Civilization, or homogeneous, like the Japanese civilization. What Huntington calls the "clash of civilizations" might be characterized by Wilkinson as a clash of cultural spheres within a single global civilization. Others point to the Crusading movement as the first step in globalization. The more conventional viewpoint is that networks of societies have expanded and shrunk since ancient times, and that the current globalized economy and culture is a product of recent European colonialism. 

History 

The notion of human history as a succession of "civilizations" is an entirely modern one. In the European Age of Discovery, emerging Modernity was put into stark contrast with the Neolithic and Mesolithic stage of the cultures of many of the peoples they encountered. Nonetheless, developments in the Neolithic stage, such as agriculture and sedentary settlement, were critical to the development of modern conceptions of civilization. 

Urban Revolution 

The Natufian culture in the Levantine corridor provides the earliest case of a Neolithic Revolution, with the planting of cereal crops attested from c. 11,000 BCE. The earliest neolithic technology and lifestyle were established first in Western Asia (for example at Göbekli Tepe, from about 9,130 BCE), later in the Yellow River and Yangtze basins in China (for example the Peiligang and Pengtoushan cultures), and from these cores spread across Eurasia. Mesopotamia is the site of the earliest civilizations developing from 7,400 years ago. This area has been evaluated by Beverley Milton-Edwards as having "inspired some of the most important developments in human history including the invention of the wheel, the building of the earliest cities and the development of written cursive script". Similar pre-civilized "neolithic revolutions" also began independently from 7,000 BCE in northwestern South America (the Caral-Supe civilization) and in Mesoamerica. The Black Sea area served as a cradle of European civilization. The site of Solnitsata – a prehistoric fortified (walled) stone settlement (prehistoric proto-city) (5500–4200 BCE) – is believed by some archaeologists to be the oldest known town in present-day Europe. The 8.2 Kiloyear Arid Event and the 5.9 Kiloyear Inter-pluvial saw the drying out of semiarid regions and a major spread of deserts. This climate change shifted the cost-benefit ratio of endemic violence between communities, which saw the abandonment of unwalled village communities and the appearance of walled cities, seen by some as a characteristic of early civilizations. 

This "urban revolution"—a term introduced by Childe in the 1930s—from the 4th millennium BCE, marked the beginning of the accumulation of transferable economic surpluses, which helped economies and cities develop. Urban revolutions were associated with the state monopoly of violence, the appearance of a warrior (or soldier) class and endemic warfare (a state of continual or frequent warfare), the rapid development of hierarchies, and the use of human sacrifice. The civilized urban revolution in turn was dependent upon the development of sedentism, the domestication of grains, plants and animals, the permanence of settlements and development of lifestyles that facilitated economies of scale and accumulation of surplus production by particular social sectors. The transition from complex cultures to civilizations, while still disputed, seems to be associated with the development of state structures, in which power was further monopolized by an elite ruling class who practiced human sacrifice. Towards the end of the Neolithic period, various elitist Chalcolithic civilizations began to rise in various "cradles" from around 3600 BCE beginning with Mesopotamia, expanding into large-scale kingdoms and empires in the course of the Bronze Age (Akkadian Empire, Indus Valley Civilization, Old Kingdom of Egypt, Neo-Sumerian Empire, Middle Assyrian Empire, Babylonian Empire, Hittite Empire, and to some degree the territorial expansions of the Elamites, Hurrians, Amorites and Ebla). Outside the Old World, development took place independently in the Pre-Columbian Americas. Urbanization in the Caral-Supe civilization in what is now coastal Peru began about 3500 BCE. In North America, the Olmec civilization emerged about 1200 BCE; the oldest known Mayan city, located in what is now Guatemala, dates to about 750 BCE. and Teotihuacan (near the modern Mexico City) was one of the largest cities in the world in 350 CE, with a population of about 125,000. 

Axial Age 

The Bronze Age collapse was followed by the Iron Age around 1200 BCE, during which a number of new civilizations emerged, culminating in a period from the 8th to the 3rd century BCE which Karl Jaspers termed the Axial Age, presented as a critical transitional phase leading to classical civilization. 

Modernity 

A major technological and cultural transition to modernity began approximately 1500 CE in Western Europe, and from this beginning new approaches to science, technology and law spread rapidly around the world, incorporating earlier cultures into the technological and industrial society of the present. 

Fall of civilizations 

Civilizations are traditionally understood as ending in one of two ways; either through incorporation into another expanding civilization (e.g. as Ancient Egypt was incorporated into Hellenistic Greek, and subsequently Roman civilizations), or by collapsing and reverting to a simpler form of living, as happens in so-called Dark Ages. There have been many explanations put forward for the collapse of civilization. Some focus on historical examples, and others on general theory. 

Ibn Khaldun's Muqaddimah influenced theories of the analysis, growth, and decline of the Islamic civilization. He suggested repeated invasions from nomadic peoples and limited development could lead to social collapse. Edward Gibbon's work The Decline and Fall of the Roman Empire is a well-known and detailed analysis of the fall of Roman civilization. Gibbon suggested the final act of the collapse of Rome was the fall of Constantinople to the Ottoman Turks in 1453 CE. For Gibbon, "The decline of Rome was the natural and inevitable effect of immoderate greatness. Prosperity ripened the principle of decay; the cause of the destruction multiplied with the extent of conquest; and, as soon as time or accident had removed the artificial supports, the stupendous fabric yielded to the pressure of its own weight. The story of the ruin is simple and obvious; and instead of inquiring why the Roman Empire was destroyed, we should rather be surprised that it has subsisted for so long". Theodor Mommsen in his History of Rome suggested Rome collapsed with the collapse of the Western Roman Empire in 476 CE and he also tended towards a biological analogy of "genesis", "growth", "senescence", "collapse" and "decay". Oswald Spengler, in his Decline of the West rejected Petrarch's chronological division, and suggested that there had been only eight "mature civilizations". Growing cultures, he argued, tend to develop into imperialistic civilizations, which expand and ultimately collapse, with democratic forms of government ushering in plutocracy and ultimately imperialism. Arnold J. Toynbee in his A Study of History suggested that there had been a much larger number of civilizations, including a small number of arrested civilizations, and that all civilizations tended to go through the cycle identified by Mommsen. The cause of the fall of a civilization occurred when a cultural elite became a parasitic elite, leading to the rise of internal and external proletariats. Joseph Tainter in The Collapse of Complex Societies suggested that there were diminishing returns to complexity, due to which, as states achieved a maximum permissible complexity, they would decline when further increases actually produced a negative return. Tainter suggested that Rome achieved this figure in the 2nd century CE. Jared Diamond in his 2005 book Collapse: How Societies Choose to Fail or Succeed suggests five major reasons for the collapse of 41 studied cultures: environmental damage, such as deforestation and soil erosion; climate change; dependence upon long-distance trade for needed resources; increasing levels of internal and external violence, such as war or invasion; and societal responses to internal and environmental problems. Peter Turchin in his Historical Dynamics and Andrey Korotayev et al. in their Introduction to Social Macrodynamics, Secular Cycles, and Millennial Trends suggest a number of mathematical models describing collapse of agrarian civilizations. For example, the basic logic of Turchin's "fiscal-demographic" model can be outlined as follows: during the initial phase of a sociodemographic cycle we observe relatively high levels of per capita production and consumption, which leads not only to relatively high population growth rates, but also to relatively high rates of surplus production. As a result, during this phase the population can afford to pay taxes without great problems, the taxes are quite easily collectible, and the population growth is accompanied by the growth of state revenues. During the intermediate phase, the increasing population growth leads to the decrease of per capita production and consumption levels, it becomes more and more difficult to collect taxes, and state revenues stop growing, whereas the state expenditures grow due to the growth of the population controlled by the state. As a result, during this phase the state starts experiencing considerable fiscal problems. During the final pre-collapse phases the overpopulation leads to further decrease of per capita production, the surplus production further decreases, state revenues shrink, but the state needs more and more resources to control the growing (though with lower and lower rates) population. Eventually this leads to famines, epidemics, state breakdown, and demographic and civilization collapse. Peter Heather argues in his book The Fall of the Roman Empire: a New History of Rome and the Barbarians that this civilization did not end for moral or economic reasons, but because centuries of contact with barbarians across the frontier generated its own nemesis by making them a more sophisticated and dangerous adversary. The fact that Rome needed to generate ever greater revenues to equip and re-equip armies that were for the first time repeatedly defeated in the field, led to the dismemberment of the Empire. Although this argument is specific to Rome, it can also be applied to the Asiatic Empire of the Egyptians, to the Han and Tang dynasties of China, to the Muslim Abbasid Caliphate and others. Bryan Ward-Perkins, in his book The Fall of Rome and the End of Civilization, argues from mostly archaeological evidence that the collapse of Roman civilization in western Europe had deleterious impacts on the living standards of the population, unlike some historians who downplay this. The collapse of complex society meant that even basic plumbing for the elite disappeared from the continent for 1,000 years. Similar impacts have been postulated for the Dark Age after the Late Bronze Age collapse in the Eastern Mediterranean, the collapse of the Maya, on Easter Island and elsewhere. Arthur Demarest argues in Ancient Maya: The Rise and Fall of a Rainforest Civilization, using a holistic perspective to the most recent evidence from archaeology, paleoecology, and epigraphy, that no one explanation is sufficient but that a series of erratic, complex events, including loss of soil fertility, drought and rising levels of internal and external violence led to the disintegration of the courts of Mayan kingdoms, which began a spiral of decline and decay. He argues that the collapse of the Maya has lessons for civilization today. Jeffrey A. McNeely has recently suggested that "a review of historical evidence shows that past civilizations have tended to over-exploit their forests, and that such abuse of important resources has been a significant factor in the decline of the over-exploiting society". Thomas Homer-Dixon considers the fall in the energy return on investments. The energy expended to energy yield ratio is central to limiting the survival of civilizations. The degree of social complexity is associated strongly, he suggests, with the amount of disposable energy environmental, economic and technological systems allow. When this amount decreases civilizations either have to access new energy sources or collapse. Feliks Koneczny in his work "On the Plurality of Civilizations" calls his study the science on civilizations. He asserts that civilizations fall not because they must or there exist some cyclical or a "biological" life span and that there stil exist two ancient civilizations – Brahmin-Hindu and Chinese – which are not ready to fall any time soon. Koneczny claimed that civilizations cannot be mixed into hybrids, an inferior civilization when given equal rights within a highly developed civilization will overcome it. One of Koneczny's claims in his study on civilizations is that "a person cannot be civilized in two or more ways" without falling into what he calls an "abcivilized state" (as in abnormal). He also stated that when two or more civilizations exist next to one another and as long as they are vital, they will be in an existential combat imposing its own "method of organizing social life" upon the other. Absorbing alien "method of organizing social life" that is civilization and giving it equal rights yields a process of decay and decomposition. 

Future 

According to political scientist Samuel P. Huntington, the 21st century will be characterized by a clash of civilizations, which he believes will replace the conflicts between nation-states and ideologies that were prominent in the 19th and 20th centuries. However, this viewpoint has been strongly challenged by others such as Edward Said, Muhammed Asadi and Amartya Sen. Ronald Inglehart and Pippa Norris have argued that the "true clash of civilizations" between the Muslim world and the West is caused by the Muslim rejection of the West's more liberal sexual values, rather than a difference in political ideology, although they note that this lack of tolerance is likely to lead to an eventual rejection of (true) democracy. In Identity and Violence Sen questions if people should be divided along the lines of a supposed "civilization", defined by religion and culture only. He argues that this ignores the many others identities that make up people and leads to a focus on differences. Cultural historian Morris Berman argues in Dark Ages America: the End of Empire that in the corporate consumerist United States, the very factors that once propelled it to greatness―extreme individualism, territorial and economic expansion, and the pursuit of material wealth―have pushed the United States across a critical threshold where collapse is inevitable. Politically associated with over-reach, and as a result of the environmental exhaustion and polarization of wealth between rich and poor, he concludes the current system is fast arriving at a situation where continuation of the existing system saddled with huge deficits and a hollowed-out economy is physically, socially, economically and politically impossible. Although developed in much more depth, Berman's thesis is similar in some ways to that of Urban Planner, Jane Jacobs who argues that the five pillars of United States culture are in serious decay: community and family; higher education; the effective practice of science; taxation and government; and the self-regulation of the learned professions. The corrosion of these pillars, Jacobs argues, is linked to societal ills such as environmental crisis, racism and the growing gulf between rich and poor. Cultural critic and author Derrick Jensen argues that modern civilization is directed towards the domination of the environment and humanity itself in an intrinsically harmful, unsustainable, and self-destructive fashion. Defending his definition both linguistically and historically, he defines civilization as "a culture... that both leads to and emerges from the growth of cities", with "cities" defined as "people living more or less permanently in one place in densities high enough to require the routine importation of food and other necessities of life". This need for civilizations to import ever more resources, he argues, stems from their over-exploitation and diminution of their own local resources. Therefore, civilizations inherently adopt imperialist and expansionist policies and, to maintain these, highly militarized, hierarchically structured, and coercion-based cultures and lifestyles. The Kardashev scale classifies civilizations based on their level of technological advancement, specifically measured by the amount of energy a civilization is able to harness. The scale is only hypothetical, but it puts energy consumption in a cosmic perspective. The Kardashev scale makes provisions for civilizations far more technologically advanced than any currently known to exist. An alternative view suggests that Kardashev’s idea of ever increasing energy consumption may itself reflect a relatively crude phase of our technological development, rather than a universal principle. Future civilization will instead be characterized by technological minimalism, in which highly advanced society seeks to maximize effectiveness while minimizing energy use. Rather than pursuing ever-increasing levels of power consumption, such civilization might focus on optimization, efficiency, and extreme miniaturization. Mastery of quantum-scale engineering could allow them to perform complex functions using only negligible amounts of energy. A sufficiently advanced technological society might also choose to separate its technological systems from the surrounding environment. Viewed from interstellar distances, such a world could appear pristine—supporting a flourishing natural biosphere, with little or no visible trace of industry or artificial modification. 

Non-human civilizations The current scientific consensus is that human beings are the only animal species with the cognitive ability to create civilizations that has emerged on Earth. A recent thought experiment, the silurian hypothesis, however, considers whether it would "be possible to detect an industrial civilization in the geological record" given the paucity of geological information about eras before the quaternary. Astronomers speculate about the existence of communicating alien intelligent civilizations within and beyond the Milky Way galaxy, usually using variants of the Drake equation. They conduct searches for such intelligences – such as for technological traces, called "technosignatures". The proposed proto-scientific field "xenoarchaeology" is concerned with the study of artifact remains of non-human civilizations to reconstruct and interpret past lives of alien societies if such get discovered and confirmed scientifically. 

See also 

Civilization state Built environment Colony Cultural invention Sociocultural evolution Sociocultural system 

Notes 

References 

Bibliography 

Further reading Gribbin, John, "Alone in the Milky Way: Why We Are Probably the Only Intelligent Life in the Galaxy", Scientific American, vol. 319, no. 3 (September 2018), pp. 94–99. "Is life likely to exist elsewhere in the [Milky Way] galaxy? Almost certainly yes, given the speed with which it appeared on Earth. Is another technological civilization likely to exist today? Almost certainly no, given the chain of circumstances that led to our existence. These considerations suggest that we are unique not just on our planet but in the whole Milky Way. And if our planet is so special, it becomes all the more important to preserve this unique world for ourselves, our descendants and the many creatures that call Earth home." (p. 99.) 

External links 

BBC on civilization Top 10 oldest civilizations 

[Music] Music is the arrangement of sound to create some combination of form, harmony, melody, rhythm, or otherwise expressive content. Music is generally agreed to be a cultural universal that is present in all human societies. Definitions of music vary widely in substance and approach. While scholars agree that music is defined by a small number of specific elements, there is no consensus as to what these necessary elements are. Music is often characterized as a highly versatile medium for expressing human creativity. Diverse activities are involved in the creation of music, and are often divided into categories of composition, improvisation, and performance. Music may be performed using a wide variety of musical instruments, including the human voice. It can also be composed, sequenced, or otherwise produced to be indirectly played mechanically or electronically, such as via a music box, barrel organ, or digital audio workstation software on a computer. Music often plays a key role in social events and religious ceremonies. The techniques of making music are often transmitted as part of a cultural tradition. Music is played in public and private contexts, highlighted at events such as festivals and concerts for various different types of ensembles. Music is used in the production of other media, such as in soundtracks to films, TV shows, operas, and video games. Listening to music is a common means of entertainment. The culture surrounding music extends into areas of academic study, journalism, philosophy, psychology, and therapy. The music industry includes songwriters, performers, sound engineers, producers, tour organizers, distributors of instruments, accessories, and publishers of sheet music and recordings. Technology facilitating the recording and reproduction of music has historically included sheet music, microphones, phonographs, and tape machines, with playback of digital music being a common use for MP3 players, CD players, and smartphones. 

Etymology and terminology 

The modern English word music came into use in the 1630s. It descends from Middle English musike, which in turn descends from Old French musique, then Latin mūsica—and ultimately from Ancient Greek mousiké technē (μουσική τέχνη), a phrase meaning "art of the Muses". The Muses were nine deities in Ancient Greek mythology who presided over the arts and sciences. They were included in tales by the earliest Western authors, Homer and Hesiod, and eventually came to be associated with music specifically. Over time, Polyhymnia would reside over music more prominently than the other muses. The Latin word musica was also the originator for both the Spanish música and French musique via spelling and linguistic adjustment, though other European terms were probably loanwords, including the Italian musica, German Musik, Dutch muziek, Norwegian musikk, Polish muzyka, and Russian muzïka. The modern Western world usually defines music as an all-encompassing term used to describe diverse genres, styles, and traditions. This is not the case worldwide, and languages such as modern Indonesian (musik) and Shona (musakazo) have recently adopted words to reflect this universal conception, as they did not have words that fit exactly the Western scope. Before Western contact in East Asia, neither Japan nor China had a single word that encompassed music in a broad sense, but culturally, they often regarded music in such a fashion. The closest word to mean music in Chinese, yue, shares a character with le, meaning joy, and originally referred to all the arts before narrowing in meaning. Africa is too diverse to make firm generalizations, but the musicologist J. H. Kwabena Nketia has emphasized African music's often inseparable connection to dance and speech in general. Some African cultures, such as the Songye people of the Democratic Republic of the Congo and the Tiv people of Nigeria, have a strong and broad conception of 'music' but no corresponding word in their native languages. Other words commonly translated as music often have more specific meanings in their respective cultures: the Hindi word for music, sangita, properly refers to art music, while the many Indigenous languages of the Americas have words for music that refer specifically to song but describe instrumental music regardless. Though the Arabic musiqi can refer to all music, it is usually used for instrumental and metric music, while khandan identifies vocal and improvised music. 

History 

Origins and prehistory 

It is often debated to what extent the origins of music will ever be understood, and there are competing theories that aim to explain it. Many scholars highlight a relationship between the origin of music and the origin of language, and there is disagreement surrounding whether music developed before, after, or simultaneously with language. A similar source of contention surrounds whether music was the intentional result of natural selection or was a byproduct spandrel of evolution. The earliest influential theory was proposed by Charles Darwin in 1871, who stated that music arose as a form of sexual selection, perhaps via mating calls. Darwin's original perspective has been heavily criticized for its inconsistencies with other sexual selection methods, though many scholars in the 21st century have developed and promoted the theory. Other theories include that music arose to assist in organizing labor, improving long-distance communication, benefiting communication with the divine, assisting in community cohesion, or as a defense to scare off predators. Prehistoric music can only be theorized based on findings from paleolithic archaeology sites. The disputed Divje Babe flute, a perforated cave bear femur, is at least 40,000 years old, though there is considerable debate surrounding whether it is truly a musical instrument or an object formed by animals. The earliest objects whose designations as musical instruments are widely accepted are bone flutes from the Swabian Jura, Germany, namely from the Geissenklösterle, Hohle Fels, and Vogelherd caves. Dated to the Aurignacian (of the Upper Paleolithic) and used by Early European modern humans, from all three caves there are eight examples, four made from the wing bones of birds and four from mammoth ivory; three of these are near complete. Three flutes from the Geissenklösterle are dated as the oldest, c. 43,150–39,370 BP. 

Antiquity 

The earliest material and representational evidence of Egyptian musical instruments dates to the Predynastic period, but the evidence is more securely attested in the Old Kingdom when harps, flutes, and double clarinets were played. Percussion instruments, lyres, and lutes were added to orchestras by the Middle Kingdom. Cymbals frequently accompanied music and dance, much as they still do in Egypt today. Egyptian folk music, including the traditional Sufi dhikr rituals, are the closest contemporary music genre to ancient Egyptian music, having preserved many of its features, rhythms and instruments. The "Hurrian Hymn to Nikkal", found on clay tablets in the ancient Syrian city of Ugarit, is the oldest surviving notated work of music, dating back to approximately 1400 BCE. Music was an important part of social and cultural life in ancient Greece and was one of the main subjects taught to children. Musical education was considered important for the development of an individual's soul. Musicians and singers played a prominent role in Greek theater, and those who received a musical education were seen as nobles and in perfect harmony (as can be read in the Republic, Plato). Mixed gender choruses performed for entertainment, celebration, and spiritual ceremonies. Instruments included the double-reed aulos and a plucked string instrument, the lyre, principally a special kind called a kithara. Music was an important part of education, and boys were taught music starting at age six. Greek musical literacy created significant musical development. Greek music theory included the Greek musical modes, which eventually became the basis for Western religious and classical music. Later, influences from the Roman Empire, Eastern Europe, and the Byzantine Empire changed Greek music. The Seikilos epitaph is the oldest surviving example of a complete musical composition, including musical notation, from anywhere in the world. The oldest surviving work written about music theory is Harmonika Stoicheia by Aristoxenus. 

Asian cultures 

Asian music covers a swath of music cultures surveyed in the articles on Arabia, Central Asia, East Asia, South Asia, and Southeast Asia. These traditions, with several reaching into antiquity, share historical ties through trade, religion and philosophical exchange, yet preserve their unique local aesthetics and performance styles. 

Indian classical music is one of the oldest musical traditions in the world. Sculptures from the Indus Valley civilization show dance and old musical instruments, like the seven-holed flute. Stringed instruments and drums have been recovered from Harappa and Mohenjo Daro by excavations carried out by Mortimer Wheeler. The Rigveda, an ancient Hindu text, has elements of present Indian music, with musical notation to denote the meter and mode of chanting. Indian classical music (marga) is monophonic, and based on a single melody line or raga rhythmically organized through talas. The poem Cilappatikaram provides information about how new scales can be formed by modal shifting of the tonic from an existing scale. Present day Hindi music was influenced by Persian traditional music and Afghan Mughals. Carnatic music, popular in the southern states, is largely devotional; the majority of the songs are addressed to the Hindu deities. There are songs emphasizing love and other social issues. 

Indonesian music has been formed since the Bronze Age culture migrated to the Indonesian archipelago in the 2nd-3rd centuries BCE. Indonesian traditional music uses percussion instruments, especially kendang and gongs. Some of them developed elaborate and distinctive instruments, such as the sasando stringed instrument on the island of Rote, the Sundanese angklung, and the complex and sophisticated Javanese and Balinese gamelan orchestras. Indonesia is the home of gong chime, a general term for a set of small, high pitched pot gongs. Gongs are usually placed in order of note, with the boss up on a string held in a low wooden frame. The most popular form of Indonesian music is gamelan, an ensemble of tuned percussion instruments that include metallophones, drums, gongs, and spike fiddles along with bamboo suling (like a flute). Chinese classical music, the traditional art or court music of China, has a history stretching over about 3,000 years. It has its own unique systems of musical notation, as well as musical tuning and pitch, musical instruments and styles or genres. Chinese music is pentatonic-diatonic, having a scale of twelve notes to an octave (5 + 7 = 12) as does European-influenced music. 

Western classical 

Early music 

The medieval music era (500 to 1400), which took place during the Middle Ages, started with the introduction of monophonic (single melodic line) chanting into Catholic Church services. Musical notation was used since ancient times in Greek culture, but in the Middle Ages, notation was first introduced by the Catholic Church, so chant melodies could be written down, to facilitate the use of the same melodies for religious music across the Catholic empire. The only European Medieval repertory that has been found, in written form, from before 800 is the monophonic liturgical plainsong chant of the Catholic Church, the central tradition of which was called Gregorian chant. Alongside these traditions of sacred and church music there existed a vibrant tradition of secular song (non-religious songs). Examples of composers from this period are Léonin, Pérotin, Guillaume de Machaut, and Walther von der Vogelweide. Renaissance music (c. 1400 to 1600) was more focused on secular themes, such as courtly love. Around 1450, the printing press was invented, which made printed sheet music much less expensive and easier to mass-produce (prior to the invention of the press, all notated music was hand-copied). The increased availability of sheet music spread musical styles quicker and across a larger area. Musicians and singers often worked for the church, courts, and towns. Church choirs grew in size, and the church remained an important patron of music. By the middle of the 15th century, composers wrote richly polyphonic sacred music, in which different melody lines were interwoven simultaneously. Prominent composers from this era include Guillaume Du Fay, Giovanni Pierluigi da Palestrina, Thomas Morley, Orlando di Lasso, and Josquin des Prez. As musical activity shifted from the church to aristocratic courts, kings, queens and princes competed for the finest composers. Many leading composers came from the Netherlands, Belgium, and France; they are called the Franco-Flemish composers. They held important positions throughout Europe, especially in Italy. Other countries with vibrant musical activity included Germany, England, and Spain. 

Common practice period 

Baroque 

The Baroque era of music took place from 1600 to 1750, coinciding with the flourishing of the Baroque artistic style in Europe. The start of the Baroque era was marked by the penning of the first operas. Polyphonic contrapuntal music (music with separate, simultaneous melodic lines) remained important during this period. German Baroque composers wrote for small ensembles including strings, brass, and woodwinds, as well as for choirs and keyboard instruments such as pipe organ, harpsichord, and clavichord. Musical complexity increased during this time. Several major musical forms were created, some of them which persisted into later periods, seeing further development. These include the fugue, the invention, the sonata, and the concerto. The late Baroque style was polyphonically complex and richly ornamented. Important composers from the Baroque era include Johann Sebastian Bach (Cello suites), George Frideric Handel (Messiah), Georg Philipp Telemann, and Antonio Vivaldi (The Four Seasons). 

Classicism 

The music of the Classical period (1730 to 1820) aimed to imitate what were seen as the key elements of the art and philosophy of Ancient Greece and Rome: the ideals of balance, proportion, and disciplined expression. Music from the Classical period has a lighter, clearer, and considerably simpler texture than the Baroque music which preceded it. The main style was homophony, where a prominent melody and a subordinate chordal accompaniment part are clearly distinct. Classical instrumental melodies tended to be almost voicelike and singable. New genres were developed, and the fortepiano, the forerunner to the modern piano, replaced the Baroque era harpsichord and pipe organ as the main keyboard instrument (though pipe organ continued to be used in sacred music, such as Masses). Importance was given to instrumental music. It was dominated by further development of musical forms initially defined in the Baroque period: the sonata, the concerto, and the symphony. Other main kinds were the trio, string quartet, serenade, and divertimento. The sonata was the most important and developed form. Although Baroque composers also wrote sonatas, the Classical style of sonata is completely distinct. All of the main instrumental forms of the Classical era, from string quartets to symphonies and concertos, were based on the structure of the sonata. The instruments used chamber music and orchestra became more standardized. In place of the basso continuo group of the Baroque era, which consisted of harpsichord, organ, or lute along with a number of bass instruments selected at the discretion of the group leader (e.g., viol, cello, theorbo, serpent), Classical chamber groups used specified, standardized instruments (e.g., a string quartet would be performed by two violins, a viola, and a cello). The practice of improvised chord-playing by the continuo keyboardist or lute player, a hallmark of Baroque music, underwent a gradual decline between 1750 and 1800. One of the most important changes made in the Classical period was the development of public concerts. The aristocracy still played a significant role in the sponsorship of concerts and compositions, but it was now possible for composers to survive without being permanent employees of queens or princes. The increasing popularity of classical music led to a growth in the number and types of orchestras. The expansion of orchestral concerts necessitated the building of large public performance spaces. Symphonic music including symphonies, musical accompaniment to ballet, and mixed vocal/instrumental genres, such as opera and oratorio, became more popular. The best known composers of Classicism are Carl Philipp Emanuel Bach, Christoph Willibald Gluck, Johann Christian Bach, Joseph Haydn, Wolfgang Amadeus Mozart, Ludwig van Beethoven, and Franz Schubert. Beethoven and Schubert are also considered to be composers in the later part of the Classical era, as it began to move towards Romanticism. 

Romanticism 

Romantic music (c. 1820 to 1900) from the 19th century had many elements in common with the Romantic styles in literature and painting of the era. Romanticism was an artistic, literary, and intellectual movement was characterized by its emphasis on emotion and individualism as well as glorification of all the past and nature. Romantic music expanded beyond the rigid styles and forms of the Classical era into more passionate, dramatic, and expressive pieces and songs. Romantic composers such as Wagner and Brahms attempted to increase emotional expression and power in their music to describe deeper truths or human feelings. With symphonic tone poems, composers tried to tell stories and evoke images or landscapes using instrumental music. Some composers promoted nationalistic pride with patriotic orchestral music inspired by folk music. The emotional and expressive qualities of music came to take precedence over tradition. Romantic composers grew in idiosyncrasy, and went further in the syncretism of exploring different art-forms in a musical context, (such as literature), history (historical figures and legends), or nature itself. Romantic love or longing was a prevalent theme in many works composed during this period. In some cases, the formal structures from the classical period continued to be used (e.g., the sonata form used in string quartets and symphonies), but these forms were expanded and altered. In many cases, new approaches were explored for existing genres, forms, and functions. Also, new forms were created that were deemed better suited to the new subject matter. Composers continued to develop opera and ballet music, exploring new styles and themes. In the years after 1800, the music developed by Ludwig van Beethoven and Franz Schubert introduced a more dramatic, expressive style. In Beethoven's case, short motifs, developed organically, came to replace melody as the most significant compositional unit (an example is the distinctive four note figure used in his Fifth Symphony). Later Romantic composers such as Pyotr Ilyich Tchaikovsky, Antonín Dvořák, and Gustav Mahler used more unusual chords and more dissonance to create dramatic tension. They generated complex and often much longer musical works. During the late Romantic period, composers explored dramatic chromatic alterations of tonality, such as extended chords and altered chords, which created new sound "colors." The late 19th century saw a dramatic expansion in the size of the orchestra, and the Industrial Revolution helped to create better instruments, creating a more powerful sound. Public concerts became an important part of well-to-do urban society. It also saw a new diversity in theatre music, including operetta, and musical comedy and other forms of musical theatre. 

20th and 21st century 

In the 19th century, a key way new compositions became known to the public was by the sales of sheet music, which middle class amateur music lovers would perform at home, on their piano or other common instruments, such as the violin. With 20th-century music, the invention of new electric technologies such as radio broadcasting and mass market availability of gramophone records meant sound recordings heard by listeners (on the radio or record player) became the main way to learn about new songs and pieces. There was a vast increase in music listening as the radio gained popularity and phonographs were used to replay and distribute music; anyone with a radio or record player could hear operas, symphonies, and big bands in their own living room. During the 19th century, the focus on sheet music had restricted access to new music to middle and upper-class people who could read music and who owned pianos and other instruments. Radios and record players allowed lower-income people, who could not afford an opera or symphony concert ticket, to hear this music. As well, people could hear music from different parts of the country, or even different parts of the world, even if they could not afford to travel to these locations. This helped to spread musical styles. The focus of art music in the 20th century was characterized by exploration of new rhythms, styles, and sounds. The horrors of World War I influenced many of the arts, including music, and composers began exploring darker, harsher sounds. Traditional music styles such as jazz and folk music were used by composers as a source of ideas for classical music. Igor Stravinsky, Arnold Schoenberg, and John Cage were influential composers in 20th-century art music. The invention of sound recording and the ability to edit music gave rise to new subgenres of classical music, including the acousmatic and musique concrète schools of electronic composition. Sound recording was a major influence on the development of popular music genres, because it enabled recordings of songs and bands to be widely distributed. The introduction of the multitrack recording system had a major influence on rock music, because it could do more than record a band's performance. Using a multitrack system, a band and their music producer could overdub many layers of instrument tracks and vocals, creating new sounds that would not be possible in a live performance. Jazz evolved and became an important genre of music over the course of the 20th century, and during the second half, rock music did the same. Jazz is an American musical artform that originated in the late 19th and early 20th centuries in African American communities in the Southern United States from a confluence of African and European music traditions. The style's West African pedigree is evident in its use of blue notes, improvisation, polyrhythms, syncopation, and the swung note. 

Rock music is a genre of popular music that developed in the 1950s from rock and roll, rockabilly, blues, and country music. The sound of rock often revolves around the electric or acoustic guitar, and it uses a strong back beat laid down by a rhythm section. Along with the guitar or keyboards, saxophone and blues-style harmonica are used as soloing instruments. In its "purest form", it "has three chords, a strong, insistent back beat, and a catchy melody." The traditional rhythm section for popular music is rhythm guitar, electric bass guitar, and drums. Some bands have keyboard instruments such as organ, piano, or, since the 1970s, analog synthesizers. In the 1980s, pop musicians began using digital synthesizers, such as the DX-7 synthesizer, electronic drum machines such as the TR-808, and synth bass devices (such as the TB-303) or synth bass keyboards. In the 1990s, an increasingly large range of computerized hardware musical devices and instruments and software (e.g. digital audio workstations) were used. In the 2020s, soft synths and computer music apps make it possible for bedroom producers to create and record types of music, such as electronic dance music, in their home, adding sampled and digital instruments and editing the recording digitally. In the 1990s, bands in genres such as nu metal began including DJs in their bands. DJs create music by manipulating recorded music, using a DJ mixer. 

Creation 

Composition 

"Composition" is the act or practice of creating a song, an instrumental music piece, a work with both singing and instruments, or another type of music. In many cultures, including Western classical music, the act of composing also includes the creation of music notation, such as a sheet music "score", which is then performed by the composer or by other singers or musicians. In popular music and traditional music, the act of composing, which is typically called songwriting, may involve the creation of a basic outline of the song, called the lead sheet, which sets out the melody, lyrics, and chord progression. In classical music, the composer typically orchestrates their own compositions, but in musical theatre and in pop music, songwriters may hire an arranger to do the orchestration. In some cases, songwriters may not use notation at all, and instead compose the song in their minds and then play or record it from memory. In jazz and popular music, notable recordings by influential performers are given the weight that written scores play in classical music. Even when music is notated relatively precisely, as in classical music, there are many decisions that a performer has to make, because notation does not specify all of the elements of music precisely. The process of deciding how to perform music that has been previously composed and notated is termed "interpretation". Different performers' interpretations of the same work of music can vary widely, in terms of the tempos that are chosen and the playing or singing style or phrasing of the melodies. Composers and songwriters who present their own music are interpreting their songs, just as much as those who perform the music of others. The standard body of choices and techniques present at a given time and a given place is referred to as performance practice, whereas interpretation is generally used to mean the individual choices of a performer. Although a musical composition often uses musical notation and has a single author, this is not always the case. A work of music can have multiple composers, which often occurs in popular music when a band collaborates to write a song, or in musical theatre, when one person writes the melodies, a second person writes the lyrics, and a third person orchestrates the songs. In some styles of music, such as the blues, a composer/songwriter may create, perform and record new songs or pieces without ever writing them down in music notation. A piece of music can also be composed with words, images, or computer programs that explain or notate how the singer or musician should create musical sounds. Examples range from avant-garde music that uses graphic notation, to text compositions such as Aus den sieben Tagen, to computer programs that select sounds for musical pieces. Music that makes heavy use of randomness and chance is called aleatoric music, and is associated with contemporary composers active in the 20th century, such as John Cage, Morton Feldman, and Witold Lutosławski. A commonly known example of chance-based music is the sound of wind chimes jingling in a breeze. The study of composition has traditionally been dominated by examination of methods and practice of Western classical music, but the definition of composition is broad enough to include the creation of popular music and traditional music songs and instrumental pieces as well as spontaneously improvised works like those of free jazz performers and African percussionists such as Ewe drummers. 

Performance 

Performance is the physical expression of music, which occurs when a song is sung or played. In classical music, a work is written in music notation by a composer and then performed once the composer is satisfied with its structure and instrumentation. However, as it gets performed, the interpretation of a song or piece can evolve and change. In classical music, instrumental performers, singers or conductors may gradually make changes to the phrasing or tempo of a piece. In popular and traditional music, the performers have more freedom to make changes to the form of a song or piece. As such, in popular and traditional music styles, even when a band plays a cover song, they can make changes such as adding a guitar solo or inserting an introduction. A performance can either be planned out and rehearsed (practiced)—which is the norm in classical music, jazz big bands, and many popular music styles—or improvised over a chord progression (a sequence of chords), which is the norm in small jazz and blues groups. Rehearsals of orchestras, concert bands and choirs are led by a conductor. Rock, blues and jazz bands are usually led by the bandleader. A rehearsal is a structured repetition of a song or piece by the performers until it can be sung or played correctly and, if it is a song or piece for more than one musician, until the parts are together from a rhythmic and tuning perspective. Many cultures have strong traditions of solo performance (in which one singer or instrumentalist performs), such as in Indian classical music, and in the Western art-music tradition. Other cultures, such as in Bali, include strong traditions of group performance. All cultures include a mixture of both, and performance may range from improvised solo playing to highly planned and organized performances such as the modern classical concert, religious processions, classical music festivals, or music competitions. Chamber music, which is music for a small ensemble with only one or a few of each type of instrument, is often seen as more intimate than large symphonic works. 

Improvisation 

Musical improvisation is the creation of spontaneous music, often within (or based on) a pre-existing harmonic framework, chord progression, or riffs. Improvisers use the notes of the chord, various scales that are associated with each chord, and chromatic ornaments and passing tones which may be neither chord tones nor from the typical scales associated with a chord. Musical improvisation can be done with or without preparation. Improvisation is a major part of some types of music, such as blues, jazz, and jazz fusion, in which instrumental performers improvise solos, melody lines, and accompaniment parts. In the Western art music tradition, improvisation was an important skill during the Baroque era and during the Classical era. In the Baroque era, performers improvised ornaments, and basso continuo keyboard players improvised chord voicings based on figured bass notation. As well, the top soloists were expected to be able to improvise pieces such as preludes. In the Classical era, solo performers and singers improvised virtuoso cadenzas during concerts. However, in the 20th and early 21st century, as "common practice" Western art music performance became institutionalized in symphony orchestras, opera houses, and ballets, improvisation has played a smaller role, as more and more music was notated in scores and parts for musicians to play. At the same time, some 20th and 21st century art music composers have increasingly included improvisation in their creative work. In Indian classical music, improvisation is a core component and an essential criterion of performances. 

Art and entertainment 

Music is composed and performed for many purposes, ranging from aesthetic pleasure, religious or ceremonial purposes, or as an entertainment product for the marketplace. When music was only available through sheet music scores, such as during the Classical and Romantic eras, music lovers would buy the sheet music of their favourite pieces and songs so that they could perform them at home on the piano. With the advent of the phonograph, records of popular songs, rather than sheet music became the dominant way that music lovers would enjoy their favourite songs. With the advent of home tape recorders in the 1980s and digital music in the 1990s, music lovers could make tapes or playlists of favourite songs and take them with them on a portable cassette player or MP3 player. Some music lovers create mix tapes of favourite songs, which serve as a "self-portrait, a gesture of friendship, prescription for an ideal party... [and] an environment consisting solely of what is most ardently loved". Amateur musicians can compose or perform music for their own pleasure and derive income elsewhere. Professional musicians are employed by institutions and organisations, including armed forces (in marching bands, concert bands, and popular music groups), religious institutions, symphony orchestras, broadcasting or film production companies, and music schools. Professional musicians sometimes work as freelancers or session musicians, seeking contracts and engagements in a variety of settings. There are often many links between amateur and professional musicians. Beginning amateur musicians take lessons with professional musicians. In community settings, advanced amateur musicians perform with professional musicians in a variety of ensembles such as community concert bands and community orchestras. A distinction is often made between music performed for a live audience and music that is performed in a studio so that it can be recorded and distributed through the music retail system or the broadcasting system. However, there are also many cases where a live performance in front of an audience is also recorded and distributed. Live concert recordings are popular in both classical music and in popular music forms such as rock, where illegally taped live concerts are prized by music lovers. In the jam band scene, live, improvised jam sessions are preferred to studio recordings. 

Notation 

Music notation typically means the written expression of music notes and rhythms on paper using symbols. When music is written down, the pitches and rhythm of the music, such as the notes of a melody, are notated. Music notation often provides instructions on how to perform the music. For example, the sheet music for a song may state the song is a "slow blues" or a "fast swing", which indicates the tempo and the genre. To read notation, a person must have an understanding of music theory, harmony and the performance practice associated with a particular song or piece's genre. Written notation varies with the style and period of music. Nowadays, notated music is produced as sheet music or, for individuals with computer scorewriter programs, as an image on a computer screen. In ancient times, music notation was put onto stone or clay tablets. To perform music from notation, a singer or instrumentalist requires an understanding of the rhythmic and pitch elements embodied in the symbols and the performance practice that is associated with a piece of music or genre. In genres requiring musical improvisation, the performer often plays from music where only the chord changes and form of the song are written, requiring the performer to have a great understanding of the music's structure, harmony, and the styles of a particular genre e.g., jazz or country music. In Western art music, the most common types of written notation are scores, which include all the music parts of an ensemble piece, and parts, which are the music notation for the individual performers or singers. In popular music, jazz, and blues, the standard musical notation is the lead sheet, which notates the melody, chords, lyrics (if it is a vocal piece), and structure of the music. Fake books are also used in jazz; they may consist of lead sheets or simply chord charts, which permit rhythm section members to improvise an accompaniment part to jazz songs. Scores and parts are also used in popular music and jazz, particularly in large ensembles such as jazz "big bands." In popular music, guitarists and electric bass players often read music notated in tablature (often abbreviated as "tab"), which indicates the location of the notes to be played on the instrument using a diagram of the guitar or bass fingerboard. Tablature was used in the Baroque era to notate music for the lute, a stringed, fretted instrument. 

Oral and aural tradition Many types of music, such as traditional blues and folk music were not written down in sheet music; instead, they were originally preserved in the memory of performers, and the songs were handed down orally, from one musician or singer to another, or aurally, in which a performer learns a song "by ear". When the composer of a song or piece is no longer known, this music is often classified as "traditional" or as a "folk song". Different musical traditions have different attitudes towards how and where to make changes to the original source material, from quite strict, to those that demand improvisation or modification to the music. A culture's history and stories may also be passed on by ear through song. 

Elements 

Music has many different fundamentals or elements. Depending on the definition of "element" being used, these can include pitch, beat or pulse, tempo, rhythm, melody, harmony, texture, style, allocation of voices, timbre or color, dynamics, expression, articulation, form, and structure. The elements of music feature prominently in the music curriculums of Australia, the UK, and the US. All three curriculums identify pitch, dynamics, timbre, and texture as elements, but the other identified elements of music are far from universally agreed upon. Below is a list of the three official versions of the "elements of music": 

Australia: pitch, timbre, texture, dynamics and expression, rhythm, form, and structure. UK: pitch, timbre, texture, dynamics, duration, tempo, structure. USA: pitch, timbre, texture, dynamics, rhythm, form, harmony, style/articulation. In relation to the UK curriculum, in 2013 the term: "appropriate musical notations" was added to their list of elements and the title of the list was changed from the "elements of music" to the "inter-related dimensions of music". The inter-related dimensions of music are listed as: pitch, duration, dynamics, tempo, timbre, texture, structure, and appropriate musical notations. The phrase "the elements of music" is used in a number of different contexts. The two most common contexts can be differentiated by describing them as the "rudimentary elements of music" and the "perceptual elements of music". 

Pitch 

Pitch is an aspect of a sound that we can hear, reflecting whether one musical sound, note, or tone is "higher" or "lower" than another musical sound, note, or tone. We can talk about the highness or lowness of pitch in the more general sense, such as the way a listener hears a piercingly high piccolo note or whistling tone as higher in pitch than a deep thump of a bass drum. We also talk about pitch in the precise sense associated with musical melodies, basslines, and chords. Precise pitch can only be determined in sounds that have a frequency that is clear and stable enough to distinguish from noise. For example, it is much easier for listeners to discern the pitch of a single note played on a piano than to try to discern the pitch of a crash cymbal that is struck. 

Melody 

A melody, also called a "tune", is a series of pitches (notes) sounding in succession (one after the other), often in a rising and falling pattern. The notes of a melody are typically created using pitch systems such as scales or modes. Melodies also often contain notes from the chords used in the song. The melodies in simple folk songs and traditional songs may use only the notes of a single scale, the scale associated with the tonic note or key of a given song. For example, a folk song in the key of C (also referred to as C major) may have a melody that uses only the notes of the C major scale (the individual notes C, D, E, F, G, A, B, and C; these are the "white notes" on a piano keyboard. On the other hand, Bebop-era jazz from the 1940s and contemporary music from the 20th and 21st centuries may use melodies with many chromatic notes (i.e., notes in addition to the notes of the major scale; on a piano, a chromatic scale would include all the notes on the keyboard, including the "white notes" and "black notes" and unusual scales, such as the whole tone scale (a whole tone scale in the key of C would contain the notes C, D, E, F♯, G♯, and A♯). A low musical line played by bass instruments, such as double bass, electric bass, or tuba, is called a bassline. 

Harmony 

Harmony refers to the "vertical" sounds of pitches in music, which means pitches that are played or sung together at the same time creates a chord. Usually, this means the notes are played at the same time, although harmony may also be implied by a melody that outlines a harmonic structure (i.e., by using melody notes that are played one after the other, outlining the notes of a chord). In music written using the system of major-minor tonality ("keys"), which includes most classical music written from 1600 to 1900 and most Western pop, rock, and traditional music, the key of a piece determines the "home note" or tonic to which the piece generally resolves, and the character (e.g. major or minor) of the scale in use. Simple classical pieces and many pop and traditional music songs are written so that all the music is in a single key. More complex Classical, pop, and traditional music songs and pieces may have two keys (and in some cases three or more keys). Classical music from the Romantic era (written from about 1820–1900) often contains multiple keys, as does jazz, especially Bebop jazz from the 1940s, in which the key or "home note" of a song may change every four bars or even every two bars. 

Rhythm 

Rhythm is the arrangement of sounds and silences in time. Meter animates time in regular pulse groupings, called measures or bars, which in Western classical, popular, and traditional music often group notes in sets of two (e.g., 2/4 time), three (e.g., 3/4 time, also known as Waltz time or 3/8 time), or four (e.g., 4/4 time). Meters are made easier to hear because songs and pieces often (but not always) place an emphasis on the first beat of each grouping. Notable exceptions exist, such as the backbeat used in much Western pop and rock, in which a song that uses a measure that consists of four beats (called 4/4 time or common time) will have accents on beats two and four, which are typically performed by the drummer on the snare drum, a loud and distinctive-sounding percussion instrument. In pop and rock, the rhythm parts of a song are played by the rhythm section, which includes chord-playing instruments (e.g., electric guitar, acoustic guitar, piano, or other keyboard instruments), a bass instrument (typically electric bass or for some styles such as jazz and bluegrass, double bass) and a drum kit player. 

Texture 

Musical texture is the overall sound of a piece of music or song. The texture of a piece or song is determined by how the melodic, rhythmic, and harmonic materials are combined in a composition, thus determining the overall nature of the sound in a piece. Texture is often described in regard to the density, or thickness, and range, or width, between lowest and highest pitches, in relative terms as well as more specifically distinguished according to the number of voices, or parts, and the relationship between these voices (see common types below). For example, a thick texture contains many 'layers' of instruments. One layer can be a string section or another brass. The thickness is affected by the amount and the richness of the instruments. Texture is commonly described according to the number of and relationship between parts or lines of music: 

monophony: a single melody (or "tune") with neither instrumental accompaniment nor a harmony part. A mother singing a lullaby to her baby would be an example. heterophony: two or more instruments or singers playing/singing the same melody, but with each performer slightly varying the rhythm or speed of the melody or adding different ornaments to the melody. Two bluegrass fiddlers playing the same traditional fiddle tune together will typically each vary the melody by some degree and each add different ornaments. polyphony: multiple independent melody lines that interweave together, which are sung or played at the same time. Choral music written in the Renaissance music era was typically written in this style. A round, which is a song such as "Row, Row, Row Your Boat", which different groups of singers all start to sing at a different time, is an example of polyphony. homophony: a clear melody supported by chordal accompaniment. Most Western popular music songs from the 19th century onward are written in this texture. Music that contains a large number of independent parts (e.g., a double concerto accompanied by 100 orchestral instruments with many interweaving melodic lines) is generally said to have a "thicker" or "denser" texture than a work with few parts (e.g., a solo flute melody accompanied by a single cello). 

Timbre 

Timbre, sometimes called "color" or "tone color" is the quality or sound of a voice or instrument. Timbre is what makes a particular musical sound different from another, even when they have the same pitch and loudness. For example, a 440 Hz A note sounds different when it is played on oboe, piano, violin, or electric guitar. Even if different players of the same instrument play the same note, their notes might sound different due to differences in instrumental technique (e.g., different embouchures), different types of accessories (e.g., mouthpieces for brass players, reeds for oboe and bassoon players), or strings made out of different materials for string players (e.g., gut strings versus steel strings). Even two instrumentalists playing the same note on the same instrument (one after the other) may sound different due to different ways of playing the instrument (e.g., two string players might hold the bow differently). The physical characteristics of sound that determine the perception of timbre include the spectrum, envelope, and overtones of a note or musical sound. For electric instruments developed in the 20th century, such as electric guitar, electric bass, and electric piano, the performer can also change the tone by adjusting equalizer controls, tone controls on the instrument, and by using electronic effects units such as distortion pedals. The tone of the electric Hammond organ is controlled by adjusting drawbars. 

Expression Expressive qualities are those elements in music that create change in music without changing the main pitches or substantially changing the rhythms of the melody and its accompaniment. Performers, including singers and instrumentalists, can add musical expression to a song or piece by adding phrasing, by adding effects such as vibrato (with voice and some instruments, such as guitar, violin, brass instruments, and woodwinds), dynamics (the loudness or softness of piece or a section of it), tempo fluctuations (e.g., ritardando or accelerando, which are, respectively slowing down and speeding up the tempo), by adding pauses or fermatas on a cadence, and by changing the articulation of the notes (e.g., making notes more pronounced or accented, by making notes more legato, which means smoothly connected, or by making notes shorter). Expression is achieved through the manipulation of pitch (such as inflection, vibrato, slides, etc.), volume (dynamics, accent, tremolo, etc.), duration (tempo fluctuations, rhythmic changes, changing note duration such as with legato and staccato, etc.), timbre (e.g. changing vocal timbre from a light to a resonant voice), and sometimes even texture (e.g. doubling the bass note for a richer effect in a piano piece). Expression therefore can be seen as a manipulation of all elements to convey "an indication of mood, spirit, character etc." and as such cannot be included as a unique perceptual element of music, although it can be considered an important rudimentary element of music. 

Form 

In music, form describes the overall structure or plan of a song or piece of music, and it describes the layout of a composition as divided into sections. In the early 20th century, Tin Pan Alley songs and Broadway musical songs were often in AABA thirty-two-bar form, in which the A sections repeated the same eight bar melody (with variation) and the B section provided a contrasting melody or harmony for eight bars. From the 1960s onward, Western pop and rock songs are often in verse-chorus form, which comprises a sequence of verse and chorus ("refrain") sections, with new lyrics for most verses and repeating lyrics for the choruses. Popular music often makes use of strophic form, sometimes in conjunction with the twelve bar blues. In the tenth edition of The Oxford Companion to Music, Percy Scholes defines musical form as "a series of strategies designed to find a successful mean between the opposite extremes of unrelieved repetition and unrelieved alteration." Examples of common forms of Western music include the fugue, the invention, sonata-allegro, canon, strophic, theme and variations, and rondo. Scholes states that European classical music had only six stand-alone forms: simple binary, simple ternary, compound binary, rondo, air with variations, and fugue (although musicologist Alfred Mann emphasized that the fugue is primarily a method of composition that has sometimes taken on certain structural conventions.) Where a piece cannot readily be broken into sectional units (though it might borrow some form from a poem, story, or programme), it is said to be through-composed. Such is often the case with a fantasia, prelude, rhapsody, etude (or study), symphonic poem, Bagatelle, impromptu, or similar composition. Professor Charles Keil classified forms and formal detail as "sectional, developmental, or variational." 

Philosophy 

The philosophy of music is the study of fundamental questions regarding music and has connections with questions in metaphysics and aesthetics. Questions include: 

What is the definition of music? (What are the necessary and sufficient conditions for classifying something as music?) What is the relationship between music and mind? What does music history reveal to us about the world? What is the connection between music and emotions? What is meaning in relation to music? In ancient times, such as with the Ancient Greeks, the aesthetics of music explored the mathematical and cosmological dimensions of rhythmic and harmonic organization. In the 18th century, focus shifted to the experience of hearing music, and thus to questions about its beauty and human enjoyment (plaisir and jouissance) of music. The origin of this philosophic shift is sometimes attributed to Alexander Gottlieb Baumgarten in the 18th century, followed by Immanuel Kant. Through their writing, the ancient term 'aesthetics', meaning sensory perception, received its present-day connotation. In the 2000s, philosophers have tended to emphasize issues besides beauty and enjoyment. For example, music's capacity to express emotion has been foregrounded. In the 20th century, important contributions were made by Peter Kivy, Jerrold Levinson, Roger Scruton, and Stephen Davies. However, many musicians, music critics, and other non-philosophers have contributed to the aesthetics of music. In the 19th century, a significant debate arose between Eduard Hanslick, a music critic and musicologist, and composer Richard Wagner regarding whether music can express meaning. Harry Partch and some other musicologists, such as Kyle Gann, have studied and tried to popularize microtonal music and the usage of alternate musical scales. Modern composers like La Monte Young, Rhys Chatham, and Glenn Branca paid much attention to a scale called just intonation. It is often thought that music has the ability to affect our emotions, intellect, and psychology; it can assuage our loneliness or incite our passions. Plato suggests in The Republic that music has a direct effect on the soul. Therefore, he proposes that in the ideal regime music would be closely regulated by the state (Book VII). In Ancient China, Confucius believed that music and rituals or rites are interconnected and harmonious with nature; he stated that music was the harmonization of heaven and earth, while the order was brought by the rites order, making them extremely crucial functions in society. 

Psychology 

Modern music psychology aims to explain and understand musical behavior and experience. Research in this field and its subfields are primarily empirical; their knowledge tends to advance on the basis of interpretations of data collected by systematic observation of and interaction with human participants. In addition to its focus on fundamental perceptions and cognitive processes, music psychology is a field of research with practical relevance for many areas, including music performance, composition, education, criticism, and therapy, as well as investigations of human aptitude, skill, intelligence, creativity, and social behavior. 

Neuroscience 

Cognitive neuroscience of music is the scientific study of brain-based mechanisms involved in the cognitive processes underlying music. These behaviours include music listening, performing, composing, reading, writing, and ancillary activities. It also is increasingly concerned with the brain basis for musical aesthetics and musical emotion. The field is distinguished by its reliance on direct observations of the brain, using such techniques as functional magnetic resonance imaging (fMRI), transcranial magnetic stimulation (TMS), magnetoencephalography (MEG), electroencephalography (EEG), and positron emission tomography (PET). 

Cognitive musicology 

Cognitive musicology is a branch of cognitive science concerned with computationally modeling musical knowledge with the goal of understanding both music and cognition. The use of computer models provides an exacting, interactive medium in which to formulate and test theories and has roots in artificial intelligence and cognitive science. Cognitive musicology investigates topics such as the parallels between language and music in the brain. Research often includes biologically inspired models of computation, such as neural networks and evolutionary programs. This field seeks to model how musical knowledge is represented, stored, perceived, performed, and generated. By using a well-structured computer environment, the systematic structures of these cognitive phenomena can be investigated. 

Psychoacoustics 

Psychoacoustics is the scientific study of sound perception. More specifically, it is the branch of science studying the psychological and physiological responses associated with sound (including speech and music). It can be further categorized as a branch of psychophysics. 

Evolutionary musicology 

Evolutionary musicology concerns the "origins of music, the question of animal song, selection pressures underlying music evolution", and "music evolution and human evolution". It seeks to understand music perception and activity in the context of evolutionary theory. Charles Darwin speculated that music may have held an adaptive advantage and functioned as a protolanguage, a view which has spawned several competing theories of music evolution. An alternate view sees music as a by-product of linguistic evolution; a type of "auditory cheesecake" that pleases the senses without providing any adaptive function. This view has been directly countered by numerous music researchers. 

Cultural effects An individual's culture or ethnicity plays a role in their music cognition, including their preferences, emotional reaction, and musical memory. Musical preferences are biased toward culturally familiar musical traditions beginning in infancy, and adults' classification of the emotion of a musical piece depends on both culturally specific and universal structural features. Additionally, individuals' musical memory abilities are greater for culturally familiar music than for culturally unfamiliar music. 

Perceptual Since the emergence of the study of psychoacoustics in the 1930s, most lists of elements of music have related more to how we hear music than how we learn to play it or study it. C.E. Seashore, in his book Psychology of Music, identified four "psychological attributes of sound". These were: "pitch, loudness, time, and timbre" (p. 3). He did not call them the "elements of music" but referred to them as "elemental components" (p. 2). Nonetheless, these elemental components link precisely with four of the most common musical elements: "Pitch" and "timbre" match exactly, "loudness" links with dynamics, and "time" links with the time-based elements of rhythm, duration, and tempo. This usage of the phrase "the elements of music" links more closely with Webster's New 20th Century Dictionary definition of an element as: "a substance which cannot be divided into a simpler form by known methods" and educational institutions' lists of elements generally align with this definition as well. Although writers of lists of "rudimentary elements of music" can vary their lists depending on their personal (or institutional) priorities, the perceptual elements of music should consist of an established (or proven) list of discrete elements which can be independently manipulated to achieve an intended musical effect. It seems at this stage that there is still research to be done in this area. A slightly different way of approaching the identification of the elements of music, is to identify the "elements of sound" as: pitch, duration, loudness, timbre, sonic texture, and spatial location, and then to define the "elements of music" as: sound, structure, and artistic intent. 

Sociological aspects 

Ethnographic studies demonstrate that music is a participatory, community-based activity. Music is experienced by individuals in a range of social settings from being alone, to attending a large concert, forming a music community, which cannot be understood as a function of individual will or accident; it includes both commercial and non-commercial participants with a shared set of common values. Musical performances take different forms in different cultures and socioeconomic milieus. In Europe and North America, there was a divide between what types of music were viewed as "high culture" and "low culture". "High culture" included Baroque, Classical, Romantic, and modern-era symphonies, concertos, and solo works, and are typically heard in formal concerts in concert halls and churches, with the audience sitting quietly. Other types of music—including jazz, blues, soul, and country—are often performed in bars, nightclubs, and theatres, where the audience may drink, dance, and cheer. Until the 20th century, the division between "high" and "low" musical forms was accepted as a valid distinction that separated out "art music" from popular music heard in bars and dance halls. Musicologists, such as David Brackett, note a "redrawing of high-low cultural-aesthetic boundaries" in the 20th century. And, "when industry and public discourses link categories of music with categories of people, they tend to conflate stereotypes with actual listening communities." Stereotypes can be based on socioeconomic standing, or social class, of the performers or audience of the different types of music. When composers introduce styles of music that break with convention, there can be strong resistance from academics and others. Late-period Beethoven string quartets, Stravinsky ballet scores, serialism, bebop, hip-hop, punk rock, and electronica were controversial and criticised, when they were first introduced. Such themes are examined in the sociology of music, sometimes called sociomusicology, which is pursued in departments of sociology, media studies, or music, and is closely related to ethnomusicology. 

Role of women 

Women have played a major role in music throughout history, as composers, songwriters, instrumental performers, singers, conductors, music scholars, music educators, music critics/music journalists, and other musical professions. In the 2010s, while women comprised a significant proportion of popular music and classical music singers, and a significant proportion of songwriters (many of them being singer-songwriters), there were few women record producers, rock critics, and rock instrumentalists. Although there have been a huge number of women composers in classical music, from the medieval period to the present day, women composers have been significantly underrepresented in the commonly performed classical music repertoire, music history textbooks, and music encyclopedias; for example, in the Concise Oxford History of Music, Clara Schumann is one of the few female composers who is mentioned. Women comprise a significant proportion of instrumental soloists in classical music and the percentage of women in orchestras is increasing. A 2015 article on concerto soloists in major Canadian orchestras, however, indicated that 84% of the soloists with the Montreal Symphony Orchestra were men. In 2012, women still made up just 6% of the top-ranked Vienna Philharmonic orchestra. Women are less common as instrumental players in popular music genres such as rock and heavy metal, although there have been a number of notable female instrumentalists and all-female bands. Women are particularly underrepresented in extreme metal genres. In the 1960s pop-music scene, "[l]ike most aspects of the...music business, [in the 1960s,] songwriting was a male-dominated field. Though there were plenty of female singers on the radio, women ...were primarily seen as consumers:... Singing was sometimes an acceptable pastime for a girl, but playing an instrument, writing songs, or producing records simply wasn't done." Young women "...were not socialized to see themselves as people who create [music]." Women are also underrepresented in orchestral conducting, music criticism/music journalism, music producing, and sound engineering. While women were discouraged from composing in the 19th century, and there are few women musicologists, women became involved in music education "...to such a degree that women dominated [this field] during the later half of the 19th century and well into the 20th century." According to Jessica Duchen, a music writer for London's The Independent, women musicians in classical music are "...too often judged for their appearances, rather than their talent" and they face pressure "...to look sexy onstage and in photos." Duchen states that while "[t]here are women musicians who refuse to play on their looks,...the ones who do tend to be more materially successful." According to the UK's Radio 3 editor, Edwina Wolstencroft, the music industry has long been open to having women in performance or entertainment roles, but women are much less likely to have positions of authority, such as being the conductor of an orchestra. In popular music, while there are many women singers recording songs, there are very few women behind the audio console acting as music producers, the individuals who direct and manage the recording process. One of the most recorded artists is Asha Bhosle, an Indian singer best known as a playback singer in Hindi cinema. 

Media and technology 

Since the 20th century, live music can be broadcast over the radio, television, or the Internet, or recorded and listened to on a CD player or MP3 player. In the early 20th century (in the late 1920s), as talking pictures emerged in the early 20th century, with their prerecorded musical tracks, an increasing number of moviehouse orchestra musicians found themselves out of work. During the 1920s, live musical performances by orchestras, pianists, and theater organists were common at first-run theaters. With the coming of the talking motion pictures, those featured performances were largely eliminated. The American Federation of Musicians (AFM) took out newspaper advertisements protesting the replacement of live musicians with mechanical playing devices. One 1929 ad that appeared in the Pittsburgh Press features an image of a can labeled "Canned Music / Big Noise Brand / Guaranteed to Produce No Intellectual or Emotional Reaction Whatever" Sometimes, live performances incorporate prerecorded sounds. For example, a disc jockey uses disc records for scratching, and some 20th-century works have a solo for an instrument or voice that is performed along with music that is prerecorded onto a tape. Some pop bands use recorded backing tracks. Computers and many keyboards can be programmed to produce and play Musical Instrument Digital Interface (MIDI) music. Audiences can also become performers by participating in karaoke, an activity of Japanese origin centered on a device that plays voice-eliminated versions of well-known songs. Most karaoke machines also have video screens that show lyrics to songs being performed; performers can follow the lyrics as they sing over the instrumental tracks. 

The advent of the Internet and widespread high-speed broadband access has transformed the experience of music, partly through the increased ease of access to recordings of music via streaming video and vastly increased choice of music for consumers. Another effect of the Internet arose with online communities and social media websites like YouTube and Facebook, a social networking service. These sites make it easier for aspiring singers and amateur bands to distribute videos of their songs, connect with other musicians, and gain audience interest. Professional musicians also use YouTube as a free publisher of promotional material. YouTube users, for example, no longer only download and listen to MP3s, but also actively create their own. According to Don Tapscott and Anthony D. Williams, in their book Wikinomics, there has been a shift from a traditional consumer role to what they call a "prosumer" role, a consumer who both creates content and consumes. Manifestations of this in music include the production of mashes, remixes, and music videos by fans. Music streaming services have further transformed music production. Streaming platforms, such as Spotify and Apple Music, mediate how music is consumed and distributed. This raises questions about how artists market their releases (i.e. as 'limited edition'), algorithmic suggestions, and affordable access to recorded music. Generative AI systems, more commonly known as Music Metacreation in music generation, are increasingly being used to create music. A more neutral term is 'Music Generation Systems' (MGSs). 

Education 

Non-institutional 

The incorporation of music into general education from preschool to post secondary education is common in North America and Europe. Involvement in playing and singing music is thought to teach basic skills such as concentration, counting, listening, and cooperation while also promoting understanding of language, improving the ability to recall information, and creating an environment more conducive to learning in other areas. In elementary schools, children often learn to play instruments such as the recorder, sing in small choirs, and learn about the history of Western art music and traditional music. Some elementary school children also learn about popular music styles. In religious schools, children sing hymns and other religious music. In secondary schools (but rarely in primary schools), students may have the opportunity to perform in some types of musical ensembles, such as choirs (a group of singers), marching bands, concert bands, jazz bands, or orchestras. In some school systems, music lessons on how to play instruments may be provided. Some students also take private music lessons after school with a singing teacher or instrument teacher. Amateur musicians typically learn basic musical rudiments (e.g., learning about musical notation for musical scales and rhythms) and beginner- to intermediate-level singing or instrument-playing techniques. At the university level, students in most arts and humanities programs can receive credit for taking a few music courses, which typically take the form of an overview course on the history of music, or a music appreciation course that focuses on listening to music and learning about different musical styles. In addition, most North American and European universities have some types of musical ensembles that students in arts and humanities are able to participate in, such as choirs, marching bands, concert bands, or orchestras. The study of Western art music is increasingly common outside of North America and Europe, such as the Indonesian Institute of the Arts in Yogyakarta, Indonesia, or the classical music programs that are available in Asian countries such as South Korea, Japan, and China. At the same time, Western universities and colleges are widening their curriculum to include music of non-Western cultures, such as the music of Africa or Bali (e.g. Gamelan music). 

Institutional 

People aiming to become professional musicians, singers, composers, songwriters, music teachers, and practitioners of other music-related professions such as music history professors, sound engineers, and so on study in specialized post-secondary programs offered by colleges, universities, and music conservatories. Some institutions that train individuals for careers in music offer training in a wide range of professions, as is the case with many of the top U.S. universities, which offer degrees in music performance (including singing and playing instruments), music history, music theory, music composition, music education (for individuals aiming to become elementary or high school music teachers), and, in some cases, conducting. On the other hand, some small colleges may only offer training in a single profession (e.g., sound recording). While most university and conservatory music programs focus on training students in classical music, there are universities and colleges that train musicians for careers as jazz or popular music musicians and composers, with notable U.S. examples including the Manhattan School of Music and the Berklee College of Music. Two schools in Canada which offer professional jazz training are McGill University and Humber College. Individuals aiming at careers in some types of music, such as heavy metal music, country music, or blues are unlikely to become professionals by completing degrees or diplomas. Instead, they typically learn about their style of music by singing or playing in bands (often beginning in amateur bands, cover bands, and tribute bands), studying recordings on DVD and the Internet, and working with already-established professionals in their style of music, either through informal mentoring or regular music lessons. Since the 2000s, the increasing popularity and availability of Internet forums and YouTube "how-to" videos have enabled singers and musicians from metal, blues, and similar genres to improve their skills. Many pop, rock, and country singers train informally with vocal coaches and voice teachers. 

Academic study 

Musicology 

Musicology, the academic study of music, is studied in universities and music conservatories. The earliest definitions from the 19th century defined three sub-disciplines of musicology: systematic musicology, historical musicology, and comparative musicology or ethnomusicology. In 2010-era scholarship, one is more likely to encounter a division into music theory, music history, and ethnomusicology. Research in musicology has often been enriched by cross-disciplinary work, for example in the field of psychoacoustics. The study of music of non-Western cultures, and cultural study of music, is called ethnomusicology. Students can pursue study of musicology, ethnomusicology, music history, and music theory through different types of degrees, including bachelor's, master's and PhD. 

Music theory 

Music theory is the study of music, generally in a highly technical manner outside of other disciplines. More broadly it refers to any study of music, usually related in some form with compositional concerns, and may include mathematics, physics, and anthropology. What is most commonly taught in beginning music theory classes are guidelines to write in the style of the common practice period, or tonal music. Theory, even of music of the common practice period, may take other forms. Musical set theory is the application of mathematical set theory to music, first applied to atonal music. Speculative music theory, contrasted with analytic music theory, is devoted to the analysis and synthesis of music materials, for example tuning systems, generally as preparation for composition. 

Zoomusicology 

Zoomusicology is the study of the music of non-human animals, or the musical aspects of sounds produced by non-human animals. As George Herzog (1941) asked, "do animals have music?" François-Bernard Mâche's Musique, mythe, nature, ou les Dauphins d'Arion (1983), a study of "ornitho-musicology" using a technique of Nicolas Ruwet's Language, musique, poésie (1972) paradigmatic segmentation analysis, shows that bird songs are organised according to a repetition-transformation principle. Jean-Jacques Nattiez (1990), argues that "in the last analysis, it is a human being who decides what is and is not musical, even when the sound is not of human origin. If we acknowledge that sound is not organised and conceptualised (that is, made to form music) merely by its producer, but by the mind that perceives it, then music is uniquely human." 

Ethnomusicology 

In the West, much of the history of music that is taught deals with the Western civilization's art music, known as classical music. The history of music in non-Western cultures ("world music" or the field of "ethnomusicology") is also taught in Western universities. This includes the documented classical traditions of Asian countries outside the influence of Western Europe, as well as the folk or indigenous music of various other cultures. Popular or folk styles of music in non-Western countries varied from culture to culture, and period to period. Different cultures emphasised different instruments, techniques, singing styles and uses for music. Music has been used for entertainment, ceremonies, rituals, religious purposes, and for practical and artistic communication. Non-Western music has also been used for propaganda purposes, as was the case with Chinese opera during the Cultural Revolution. There is a host of music classifications for non-Western music, many of which are caught up in the argument over the definition of music. Among the largest of these is the division between classical music (or "art" music), and popular music (or commercial music – including non-Western styles of rock, country, and pop music-related styles). Some genres do not fit neatly into one of these "big two" classifications, (such as folk music, world music, or jazz-related music). As world cultures have come into greater global contact, their indigenous musical styles have often merged with other styles, which produces new styles. For example, the United States bluegrass style contains elements from Anglo-Irish, Scottish, Irish, German, and African instrumental and vocal traditions, which were able to fuse in the United States' multi-ethnic "melting pot" society. Some types of world music contain a mixture of non-Western indigenous styles with Western pop music elements. Genres of music are determined as much by tradition and presentation as by the actual music. Some works, like George Gershwin's Rhapsody in Blue, are claimed by both jazz and classical music, while Gershwin's Porgy and Bess and Leonard Bernstein's West Side Story are claimed by both opera and the Broadway musical tradition. Many music festivals for non-Western music include bands and singers from a particular musical genre, such as world music. Indian music, for example, is one of the oldest and longest living types of music, and is still widely heard and performed in South Asia, as well as internationally (especially since the 1960s). Indian music has mainly three forms of classical music, Hindustani, Carnatic, and Dhrupad styles. It has also a large repertoire of styles, which involve only percussion music such as the talavadya performances famous in South India. 

Therapy 

Music therapy is an interpersonal process in which a trained therapist uses music and all of its facets—physical, emotional, mental, social, aesthetic, and spiritual—to help clients to improve or maintain their health. In some instances, the client's needs are addressed directly through music; in others they are addressed through the relationships that develop between the client and therapist. Music therapy is used with individuals of all ages and with a variety of conditions, including: psychiatric disorders, medical problems, physical disabilities, sensory impairments, developmental disabilities, substance abuse issues, communication disorders, interpersonal problems, and aging. It is also used to improve learning, build self-esteem, reduce stress, support physical exercise, and facilitate a host of other health-related activities. Music therapists may encourage clients to sing, play instruments, create songs, or do other musical activities. In the 10th century, the philosopher Al-Farabi described how vocal music can stimulate the feelings and souls of listeners. Music has long been used to help people deal with their emotions. In the 17th century, the scholar Robert Burton's The Anatomy of Melancholy argued that music and dance were critical in treating mental illness, especially melancholia. He noted that music has an "excellent power ...to expel many other diseases" and he called it "a sovereign remedy against despair and melancholy." He pointed out that in Antiquity, Canus, a Rhodian fiddler, used music to "make a melancholy man merry, ...a lover more enamoured, a religious man more devout." In the Ottoman Empire, mental illnesses were treated with music. In November 2006, Michael J. Crawford and his colleagues also found that music therapy helped schizophrenic patients. 

See also 

Glossary of music terminology List of musicology topics Lists of musicians Music and emotion Music archaeology – Interdisciplinary study field Music history – Academic field Music-specific disorders – Disorders relating to the perception of music 

References 

Notes 

Citations 

Sources 

Further reading 

External links 

Grove Music Online – online version of The New Grove Dictionary of Music and Musicians. All ten volumes of the Garland Encyclopedia of World Music (subscription required) Dolmetsch free online music dictionary, complete, with references to a list of specialised music dictionaries (by continent, by instrument, by genre, etc.) Some books on music by Carl Van Vechten (1880–1964) 

[Philosophy] Philosophy (from Ancient Greek philosophía, lit. 'love of wisdom') is a systematic study of general and fundamental questions concerning topics like existence, knowledge, mind, reason, language, and value. It is a rational and critical inquiry that reflects on its methods and assumptions. Historically, many of the individual sciences, such as physics and psychology, formed part of philosophy. However, they are considered separate academic disciplines in the modern sense of the term. Influential traditions in the history of philosophy include Western, Arabic–Persian, Indian, and Chinese philosophy. Western philosophy originated in Ancient Greece and covers a wide area of philosophical subfields. A central topic in Arabic–Persian philosophy is the relation between reason and revelation. Indian philosophy combines the spiritual problem of how to reach enlightenment with the exploration of the nature of reality and the ways of arriving at knowledge. Chinese philosophy focuses principally on practical issues about right social conduct, government, and self-cultivation. Major branches of philosophy are epistemology, ethics, logic, and metaphysics. Epistemology studies what knowledge is and how to acquire it. Ethics investigates moral principles and what constitutes right conduct. Logic is the study of correct reasoning and explores how good arguments can be distinguished from bad ones. Metaphysics examines the most general features of reality, existence, objects, and properties. Other subfields are aesthetics, philosophy of language, philosophy of mind, philosophy of mathematics, philosophy of science, philosophy of religion, philosophy of history, and political philosophy. Within each branch, there are competing schools of philosophy that promote different principles, theories, or methods. Philosophers use a great variety of methods to arrive at philosophical knowledge. They include conceptual analysis, reliance on common sense and intuitions, use of thought experiments, analysis of ordinary language, description of experience, and critical questioning. Many scientific academic disciplines have associated philosophical subfields, like philosophy of physics and philosophy of biology, examining their fundamental concepts, methods, assumptions and implications. Philosophy is related to and informs many other fields, such as law, business, and journalism, providing an interdisciplinary perspective and addressing their ethical questions. 

Etymology The word philosophy comes from the Ancient Greek term φιλοσοφία (philosophía), meaning 'love of wisdom', from φίλος (phílos, 'loving, friend of') and σοφία (sophía) 'wisdom'. Some sources say that the term was coined by the pre-Socratic philosopher Pythagoras, but this is not certain. The word entered the English language primarily from Old French and Anglo-Norman starting around 1175 CE. The French philosophie is itself a borrowing from the Latin philosophia. The term philosophy acquired the meanings of "advanced study of the speculative subjects (logic, ethics, physics, and metaphysics)", "deep wisdom consisting of love of truth and virtuous living", "profound learning as transmitted by the ancient writers", and "the study of the fundamental nature of knowledge, reality, and existence, and the basic limits of human understanding". Before the modern age, the term philosophy was used in a wide sense. It included most forms of rational inquiry, such as the individual sciences, as its subdisciplines. For instance, natural philosophy was a major branch of philosophy. This branch of philosophy encompassed a wide range of fields, including disciplines like physics, chemistry, and biology. An example of this usage is the 1687 book Philosophiæ Naturalis Principia Mathematica by Isaac Newton. This book referred to natural philosophy in its title, but it is today considered a book of physics. The meaning of philosophy changed toward the end of the modern period when it acquired the more narrow meaning common today. In this new sense, the term is mainly associated with disciplines like metaphysics, epistemology, and ethics. Among other topics, it covers the rational study of reality, knowledge, and values. It is distinguished from other disciplines of rational inquiry such as the empirical sciences and mathematics. 

Conceptions of philosophy 

General conception The practice of philosophy is characterized by several general features: it is a form of rational inquiry, it aims to be systematic, and it tends to critically reflect on its own methods and presuppositions. It requires attentively thinking long and carefully about the provocative, vexing, and enduring problems central to the human condition. The philosophical pursuit of wisdom involves asking general and fundamental questions. It often does not result in straightforward answers but may help a person to better understand the topic, examine their life, dispel confusion, and overcome prejudices and self-deceptive ideas associated with common sense. For example, Socrates stated that "the unexamined life is not worth living" to highlight the role of philosophical inquiry in understanding one's own existence. And according to Bertrand Russell, "the man who has no tincture of philosophy goes through life imprisoned in the prejudices derived from common sense, from the habitual beliefs of his age or his nation, and from convictions which have grown up in his mind without the cooperation or consent of his deliberate reason." 

Academic definitions 

Attempts to provide more precise definitions of philosophy are controversial and are studied in metaphilosophy. Some approaches argue that there is a set of essential features shared by all parts of philosophy. Others see only weaker family resemblances or contend that it is merely an empty blanket term. Precise definitions are often only accepted by theorists belonging to a certain philosophical movement and are revisionistic according to Søren Overgaard et al. in that many presumed parts of philosophy would not deserve the title "philosophy" if they were true. Some definitions characterize philosophy in relation to its method, like pure reasoning. Others focus on its topic; for example, as the study of the biggest patterns of the world as a whole or as the attempt to answer the big questions. Such an approach is pursued by Immanuel Kant, who holds that the task of philosophy is united by four questions: "What can I know?", "What should I do?", "What may I hope?", and "What is the human being?" Both approaches have the problem that they are usually either too wide, by including non-philosophical disciplines, or too narrow, by excluding some philosophical sub-disciplines. Many definitions of philosophy emphasize its intimate relation to science. In this sense, philosophy is sometimes understood as a proper science in its own right. According to some naturalistic philosophers, such as W. V. O. Quine, philosophy is an empirical yet abstract science that is concerned with wide-ranging empirical patterns instead of particular observations. Science-based definitions usually face the problem of explaining why philosophy in its long history has not progressed to the same extent or in the same way as the sciences. This problem is avoided by seeing philosophy as an immature or provisional science whose subdisciplines cease to be philosophy once they have fully developed. In this sense, philosophy is sometimes described as "the midwife of the sciences". Other definitions focus on the contrast between science and philosophy. A common theme among many such conceptions is that philosophy is concerned with meaning, understanding, or the clarification of language. According to one view, philosophy is conceptual analysis, which involves finding the necessary and sufficient conditions for the application of concepts. Another definition characterizes philosophy as thinking about thinking to emphasize its self-critical, reflective nature. A further approach presents philosophy as a linguistic therapy. According to Ludwig Wittgenstein, for instance, philosophy aims at dispelling misunderstandings to which humans are susceptible due to the confusing structure of ordinary language. Phenomenologists, such as Edmund Husserl, characterize philosophy as a "rigorous science" investigating essences. They practice a radical suspension of theoretical assumptions about reality to get back to the "things themselves"; that is, as originally given in experience. They contend that this base-level of experience provides the foundation for higher-order theoretical knowledge and that one needs to understand the former to understand the latter. An early approach found in ancient Greek and Roman philosophy is that philosophy is the spiritual practice of developing one's rational capacities. This practice is an expression of the philosopher's love of wisdom and has the aim of improving one's well-being by leading a reflective life. For example, the Stoics saw philosophy as an exercise to train the mind and thereby achieve eudaimonia and flourish in life. 

History 

As a discipline, the history of philosophy aims to provide a systematic and chronological exposition of philosophical concepts and doctrines. Some theorists see it as a part of intellectual history, but it also investigates questions not covered by intellectual history such as whether the theories of past philosophers are true and have remained philosophically relevant. The history of philosophy is primarily concerned with theories based on rational inquiry and argumentation; some historians understand it in a looser sense that includes myths, religious teachings, and proverbial lore. Influential traditions in the history of philosophy include Western, Arabic–Persian, Indian, and Chinese philosophy. Other philosophical traditions are Japanese philosophy, Latin American philosophy, and African philosophy. 

Western 

Western philosophy originated in Ancient Greece in the 6th century BCE with the pre-Socratic philosophers, such as Thales of Miletus (c. 626 – c. 545 BCE). They attempted to provide rational explanations of the cosmos as a whole. The philosophy following them was shaped by Socrates (469–399 BCE), Plato (427–347 BCE), and Aristotle (384–322 BCE). They expanded the range of topics to questions like how people should act, how to arrive at knowledge, and what the nature of reality and mind is. The later part of the ancient period was marked by the emergence of philosophical movements, for example, Epicureanism, Stoicism, Skepticism, and Neoplatonism. The medieval period started in the 5th century CE. Its focus was on religious topics and many thinkers used ancient philosophy to explain and further elaborate Christian doctrines. The Renaissance period started in the 14th century and saw a renewed interest in schools of ancient philosophy, in particular Platonism. Humanism also emerged in this period. The modern period started in the 17th century. One of its central concerns was how philosophical and scientific knowledge are created. Specific importance was given to the role of reason and sensory experience. Many of these innovations were used in the Enlightenment movement to challenge traditional authorities. Several attempts to develop comprehensive systems of philosophy were made in the 19th century, for instance, by German idealism and Marxism. Influential developments in 20th-century philosophy were the emergence and application of formal logic, the focus on the role of language as well as pragmatism, and movements in continental philosophy like phenomenology, existentialism, and post-structuralism. The 20th century saw a rapid expansion of academic philosophy in terms of the number of philosophical publications and philosophers working at academic institutions. There was also a noticeable growth in the number of female philosophers, but they still remained underrepresented. 

Arabic–Persian 

Arabic–Persian philosophy arose in the early 9th century CE as a response to discussions in the Islamic theological tradition. Its classical period lasted until the 12th century CE and was strongly influenced by ancient Greek philosophers. It employed their ideas to elaborate and interpret the teachings of the Quran. Al-Kindi (801–873 CE) is usually regarded as the first philosopher of this tradition. He translated and interpreted many works of Aristotle and Neoplatonists in his attempt to show that there is a harmony between reason and faith. Avicenna (980–1037 CE) also followed this goal and developed a comprehensive philosophical system to provide a rational understanding of reality encompassing science, religion, and mysticism. Al-Ghazali (1058–1111 CE) was a strong critic of the idea that reason can arrive at a true understanding of reality and God. He formulated a detailed critique of philosophy and tried to assign philosophy a more limited place besides the teachings of the Quran and mystical insight. Following Al-Ghazali and the end of the classical period, the influence of philosophical inquiry waned. Mulla Sadra (1571–1636 CE) is often regarded as one of the most influential philosophers of the subsequent period. The increasing influence of Western thought and institutions in the 19th and 20th centuries gave rise to the intellectual movement of Islamic modernism, which aims to understand the relation between traditional Islamic beliefs and modernity. 

Indian 

One of the distinguishing features of Indian philosophy is that it integrates the exploration of the nature of reality, the ways of arriving at knowledge, and the spiritual question of how to reach enlightenment. It started in the second and first millennia BCE, when the Vedas and Upanishads were composed. They are the foundational scriptures of Hinduism and contemplate issues concerning the relation between the self and ultimate reality as well as the question of how souls are reborn based on their past actions. This period also saw the emergence of non-Vedic teachings, like Buddhism and Jainism. Buddhism was founded by Gautama Siddhartha (563–483 BCE), who challenged the Vedic idea of a permanent self and proposed a path to liberate oneself from suffering. Jainism was founded by Mahavira (599–527 BCE), who emphasized non-violence as well as respect toward all forms of life. The subsequent classical period started roughly 200 BCE and was characterized by the emergence of the six orthodox schools of Hinduism: Nyāyá, Vaiśeṣika, Sāṃkhya, Yoga, Mīmāṃsā, and Vedanta. The school of Advaita Vedanta developed later in this period. It was systematized by Adi Shankara (c. 700–750 CE), who held that everything is one and that the impression of a universe consisting of many distinct entities is an illusion. A slightly different perspective was defended by Ramanuja (1017–1137 CE), who founded the school of Vishishtadvaita Vedanta and argued that individual entities are real as aspects or parts of the underlying unity. He also helped to popularize the Bhakti movement, which taught devotion toward the divine as a spiritual path and lasted until the 17th to 18th centuries CE. The modern period began roughly 1800 CE and was shaped by encounters with Western thought. Philosophers tried to formulate comprehensive systems to harmonize diverse philosophical and religious teachings. For example, Swami Vivekananda (1863–1902 CE) used the teachings of Advaita Vedanta to argue that all the different religions are valid paths toward the one divine. 

Chinese 

Chinese philosophy is particularly interested in practical questions associated with right social conduct, government, and self-cultivation. Many schools of thought emerged in the 6th century BCE in competing attempts to resolve the political turbulence of that period. The most prominent among them were Confucianism and Daoism. Confucianism was founded by Confucius (551–479 BCE). It focused on different forms of moral virtues and explored how they lead to harmony in society. Daoism was founded by Laozi (6th century BCE) and examined how humans can live in harmony with nature by following the Dao or the natural order of the universe. Other influential early schools of thought were Mohism, which developed an early form of altruistic consequentialism, and Legalism, which emphasized the importance of a strong state and strict laws. Buddhism was introduced to China in the 1st century CE and diversified into new forms of Buddhism. Starting in the 3rd century CE, the school of Xuanxue emerged. It interpreted earlier Daoist works with a specific emphasis on metaphysical explanations. Neo-Confucianism developed in the 11th century CE. It systematized previous Confucian teachings and sought a metaphysical foundation of ethics. The modern period in Chinese philosophy began in the early 20th century and was shaped by the influence of and reactions to Western philosophy. The emergence of Chinese Marxism—which focused on class struggle, socialism, and communism—resulted in a significant transformation of the political landscape. Another development was the emergence of New Confucianism, which aims to modernize and rethink Confucian teachings to explore their compatibility with democratic ideals and modern science. 

Other traditions Traditional Japanese philosophy assimilated and synthesized ideas from different traditions, including the indigenous Shinto religion and Chinese and Indian thought in the forms of Confucianism and Buddhism, both of which entered Japan in the 6th and 7th centuries. Its practice is characterized by active interaction with reality rather than disengaged examination. Neo-Confucianism became an influential school of thought in the 16th century and the following Edo period and prompted a greater focus on language and the natural world. The Kyoto School emerged in the 20th century and integrated Eastern spirituality with Western philosophy in its exploration of concepts like absolute nothingness (zettai-mu), place (basho), and the self. Latin American philosophy in the pre-colonial period was practiced by indigenous civilizations and explored questions concerning the nature of reality and the role of humans. It has similarities to indigenous North American philosophy, which covered themes such as the interconnectedness of all things. Latin American philosophy during the colonial period, starting around 1550, was dominated by religious philosophy in the form of scholasticism. Influential topics in the post-colonial period were positivism, the philosophy of liberation, and the exploration of identity and culture. Early African philosophy was primarily conducted and transmitted orally. It focused on community, morality, and ancestral ideas, encompassing folklore, wise sayings, religious ideas, and philosophical concepts like Ubuntu. Systematic African philosophy emerged at the beginning of the 20th century. It discusses topics such as ethnophilosophy, négritude, pan-Africanism, Marxism, postcolonialism, the role of cultural identity, relativism, African epistemology, and the critique of Eurocentrism. 

Core branches 

Philosophical questions can be grouped into several branches. These groupings allow philosophers to focus on a set of similar topics and interact with other thinkers who are interested in the same questions. Epistemology, ethics, logic, and metaphysics are sometimes listed as the main branches. There are many other subfields besides them and the different divisions are neither exhaustive nor mutually exclusive. For example, political philosophy, ethics, and aesthetics are sometimes linked under the general heading of value theory as they investigate normative or evaluative aspects. Furthermore, philosophical inquiry sometimes overlaps with other disciplines in the natural and social sciences, religion, and mathematics. 

Epistemology 

Epistemology is the branch of philosophy that studies knowledge. It is also known as theory of knowledge and aims to understand what knowledge is, how it arises, what its limits are, and what value it has. It further examines the nature of truth, belief, justification, and rationality. Some of the questions addressed by epistemologists include "By what method(s) can one acquire knowledge?"; "How is truth established?"; and "Can we prove causal relations?" Epistemology is primarily interested in declarative knowledge or knowledge of facts, like knowing that Princess Diana died in 1997. But it also investigates practical knowledge, such as knowing how to ride a bicycle, and knowledge by acquaintance, for example, knowing a celebrity personally. One area in epistemology is the analysis of knowledge. It assumes that declarative knowledge is a combination of different parts and attempts to identify what those parts are. An influential theory in this area claims that knowledge has three components: it is a belief that is justified and true. This theory is controversial and the difficulties associated with it are known as the Gettier problem. Alternative views state that knowledge requires additional components, like the absence of luck; different components, like the manifestation of cognitive virtues instead of justification; or they deny that knowledge can be analyzed in terms of other phenomena. Another area in epistemology asks how people acquire knowledge. Often-discussed sources of knowledge are perception, introspection, memory, inference, and testimony. According to empiricists, all knowledge is based on some form of experience. Rationalists reject this view and hold that some forms of knowledge, like innate knowledge, are not acquired through experience. The regress problem is a common issue in relation to the sources of knowledge and the justification they offer. It is based on the idea that beliefs require some kind of reason or evidence to be justified. The problem is that the source of justification may itself be in need of another source of justification. This leads to an infinite regress or circular reasoning. Foundationalists avoid this conclusion by arguing that some sources can provide justification without requiring justification themselves. Another solution is presented by coherentists, who state that a belief is justified if it coheres with other beliefs of the person. Many discussions in epistemology touch on the topic of philosophical skepticism, which raises doubts about some or all claims to knowledge. These doubts are often based on the idea that knowledge requires absolute certainty and that humans are unable to acquire it. 

Ethics 

Ethics, also known as moral philosophy, studies what constitutes right conduct. It is also concerned with the moral evaluation of character traits and institutions. It explores what the standards of morality are and how to live a good life. Philosophical ethics addresses such basic questions as "Are moral obligations relative?"; "Which has priority: well-being or obligation?"; and "What gives life meaning?" The main branches of ethics are meta-ethics, normative ethics, and applied ethics. Meta-ethics asks abstract questions about the nature and sources of morality. It analyzes the meaning of ethical concepts, like right action and obligation. It also investigates whether ethical theories can be true in an absolute sense and how to acquire knowledge of them. Normative ethics encompasses general theories of how to distinguish between right and wrong conduct. It helps guide moral decisions by examining what moral obligations and rights people have. Applied ethics studies the consequences of the general theories developed by normative ethics in specific situations, for example, in the workplace or for medical treatments. Within contemporary normative ethics, consequentialism, deontology, and virtue ethics are influential schools of thought. Consequentialists judge actions based on their consequences. One such view is utilitarianism, which argues that actions should increase overall happiness while minimizing suffering. Deontologists judge actions based on whether they follow moral duties, such as abstaining from lying or killing. According to them, what matters is that actions are in tune with those duties and not what consequences they have. Virtue theorists judge actions based on how the moral character of the agent is expressed. According to this view, actions should conform to what an ideally virtuous agent would do by manifesting virtues like generosity and honesty. 

Logic 

Logic is the study of correct reasoning. It aims to understand how to distinguish good from bad arguments. It is usually divided into formal and informal logic. Formal logic uses artificial languages with a precise symbolic representation to investigate arguments. In its search for exact criteria, it examines the structure of arguments to determine whether they are correct or incorrect. Informal logic uses non-formal criteria and standards to assess the correctness of arguments. It relies on additional factors such as content and context. Logic examines a variety of arguments. Deductive arguments are mainly studied by formal logic. An argument is deductively valid if the truth of its premises ensures the truth of its conclusion. Deductively valid arguments follow a rule of inference, like modus ponens, which has the following logical form: "p; if p then q; therefore q". An example is the argument "today is Sunday; if today is Sunday then I don't have to go to work today; therefore I don't have to go to work today". The premises of non-deductive arguments also support their conclusion, although this support does not guarantee that the conclusion is true. One form is inductive reasoning. It starts from a set of individual cases and uses generalization to arrive at a universal law governing all cases. An example is the inference that "all ravens are black" based on observations of many individual black ravens. Another form is abductive reasoning. It starts from an observation and concludes that the best explanation of this observation must be true. This happens, for example, when a doctor diagnoses a disease based on the observed symptoms. Logic also investigates incorrect forms of reasoning. They are called fallacies and are divided into formal and informal fallacies based on whether the source of the error lies only in the form of the argument or also in its content and context. 

Metaphysics 

Metaphysics is the study of the most general features of reality, such as existence, objects and their properties, wholes and their parts, space and time, events, and causation. There are disagreements about the precise definition of the term and its meaning has changed throughout the ages. Metaphysicians attempt to answer basic questions including "Why is there something rather than nothing?"; "Of what does reality ultimately consist?"; and "Are humans free?" Metaphysics is sometimes divided into general metaphysics and specific or special metaphysics. General metaphysics investigates being as such. It examines the features that all entities have in common. Specific metaphysics is interested in different kinds of being, the features they have, and how they differ from one another. An important area in metaphysics is ontology. Some theorists identify it with general metaphysics. Ontology investigates concepts like being, becoming, and reality. It studies the categories of being and asks what exists on the most fundamental level. Another subfield of metaphysics is philosophical cosmology. It is interested in the essence of the world as a whole. It asks questions including whether the universe has a beginning and an end and whether it was created by something else. A key topic in metaphysics concerns the question of whether reality only consists of physical things like matter and energy. Alternative suggestions are that mental entities (such as souls and experiences) and abstract entities (such as numbers) exist apart from physical things. Another topic in metaphysics concerns the problem of identity. One question is how much an entity can change while still remaining the same entity. According to one view, entities have essential and accidental features. They can change their accidental features but they cease to be the same entity if they lose an essential feature. A central distinction in metaphysics is between particulars and universals. Universals, like the color red, can exist at different locations at the same time. This is not the case for particulars including individual persons or specific objects. Other metaphysical questions are whether the past fully determines the present and what implications this would have for the existence of free will. 

Other major branches 

There are many other subfields of philosophy besides its core branches. Some of the most prominent are aesthetics, philosophy of language, philosophy of mind, philosophy of religion, philosophy of science, and political philosophy. Aesthetics in the philosophical sense is the field that studies the nature and appreciation of beauty and other aesthetic properties, like the sublime. Although it is often treated together with the philosophy of art, aesthetics is a broader category that encompasses other aspects of experience, such as natural beauty. In a more general sense, aesthetics is "critical reflection on art, culture, and nature". A key question in aesthetics is whether beauty is an objective feature of entities or a subjective aspect of experience. Aesthetic philosophers also investigate the nature of aesthetic experiences and judgments. Further topics include the essence of works of art and the processes involved in creating them. The philosophy of language studies the nature and function of language. It examines the concepts of meaning, reference, and truth. It aims to answer questions such as how words are related to things and how language affects human thought and understanding. It is closely related to the disciplines of logic and linguistics. The philosophy of language rose to particular prominence in the early 20th century in analytic philosophy due to the works of Frege and Russell. One of its central topics is to understand how sentences get their meaning. There are two broad theoretical camps: those emphasizing the formal truth conditions of sentences and those investigating circumstances that determine when it is suitable to use a sentence, the latter of which is associated with speech act theory. The philosophy of mind studies the nature of mental phenomena and how they are related to the physical world. It aims to understand different types of conscious and unconscious mental states, like beliefs, desires, intentions, feelings, sensations, and free will. An influential intuition in the philosophy of mind is that there is a distinction between the inner experience of objects and their existence in the external world. The mind-body problem is the problem of explaining how these two types of thing—mind and matter—are related. The main traditional responses are materialism, which assumes that matter is more fundamental; idealism, which assumes that mind is more fundamental; and dualism, which assumes that mind and matter are distinct types of entities. In contemporary philosophy, another common view is functionalism, which understands mental states in terms of the functional or causal roles they play. The mind-body problem is closely related to the hard problem of consciousness, which asks how the physical brain can produce qualitatively subjective experiences. The philosophy of religion investigates the basic concepts, assumptions, and arguments associated with religion. It critically reflects on what religion is, how to define the divine, and whether one or more gods exist. It also includes the discussion of worldviews that reject religious doctrines. Further questions addressed by the philosophy of religion are: "How are we to interpret religious language, if not literally?"; "Is divine omniscience compatible with free will?"; and, "Are the great variety of world religions in some way compatible in spite of their apparently contradictory theological claims?" It includes topics from nearly all branches of philosophy. It differs from theology since theological debates typically take place within one religious tradition, whereas debates in the philosophy of religion transcend any particular set of theological assumptions. The philosophy of science examines the fundamental concepts, assumptions, and problems associated with science. It reflects on what science is and how to distinguish it from pseudoscience. It investigates the methods employed by scientists, how their application can result in knowledge, and on what assumptions they are based. It also studies the purpose and implications of science. Some of its questions are "What counts as an adequate explanation?"; "Is a scientific law anything more than a description of a regularity?"; and "Can some special sciences be explained entirely in the terms of a more general science?" It is a vast field that is commonly divided into the philosophy of the natural sciences and the philosophy of the social sciences, with further subdivisions for each of the individual sciences under these headings. How these branches are related to one another is also a question in the philosophy of science. Many of its philosophical issues overlap with the fields of metaphysics or epistemology. Political philosophy is a branch of philosophy that aims to seek knowledge on the essence of politics. It examines the basic concepts, assumptions, and arguments in the field of politics. It investigates the nature and purpose of government and compares its different forms. It further asks under what circumstances the use of political power is legitimate, rather than a form of simple violence. In this regard, it is concerned with the distribution of political power, social and material goods, and legal rights. Other topics are justice, liberty, equality, sovereignty, and nationalism. Political philosophy involves a general inquiry into normative matters and differs in this respect from political science, which aims to provide empirical descriptions of actually existing states. Political philosophy is often treated as a subfield of ethics. Influential schools of thought in political philosophy are liberalism, conservativism, socialism, and anarchism. 

Methods 

Methods of philosophy are ways of conducting philosophical inquiry. They include techniques for arriving at philosophical knowledge and justifying philosophical claims as well as principles used for choosing between competing theories. A great variety of methods have been employed throughout the history of philosophy. Many of them differ significantly from the methods used in the natural sciences in that they do not use experimental data obtained through measuring equipment. The choice of one's method usually has important implications both for how philosophical theories are constructed and for the arguments cited for or against them. This choice is often guided by epistemological considerations about what constitutes philosophical evidence. Methodological disagreements can cause conflicts among philosophical theories or about the answers to philosophical questions. The discovery of new methods has often had important consequences both for how philosophers conduct their research and for what claims they defend. Some philosophers engage in most of their theorizing using one particular method while others employ a wider range of methods based on which one fits the specific problem investigated best. Conceptual analysis is a common method in analytic philosophy. It aims to clarify the meaning of concepts by analyzing them into their component parts. Another method often employed in analytic philosophy is based on common sense. It starts with commonly accepted beliefs and tries to draw unexpected conclusions from them, which it often employs in a negative sense to criticize philosophical theories that are too far removed from how the average person sees the issue. It is similar to how ordinary language philosophy approaches philosophical questions by investigating how ordinary language is used. 

Various methods in philosophy give particular importance to intuitions, that is, non-inferential impressions about the correctness of specific claims or general principles. For example, they play an important role in thought experiments, which employ counterfactual thinking to evaluate the possible consequences of an imagined situation. These anticipated consequences can then be used to confirm or refute philosophical theories. The method of reflective equilibrium also employs intuitions. It seeks to form a coherent position on a certain issue by examining all the relevant beliefs and intuitions, some of which often have to be deemphasized or reformulated to arrive at a coherent perspective. Pragmatists stress the significance of concrete practical consequences for assessing whether a philosophical theory is true. According to the pragmatic maxim as formulated by Charles Sanders Peirce, the idea a person has of an object is nothing more than the totality of practical consequences they associate with this object. Pragmatists have also used this method to expose disagreements as merely verbal, that is, to show they make no genuine difference on the level of consequences. Phenomenologists seek knowledge of the realm of appearance and the structure of human experience. They insist upon the first-personal character of all experience and proceed by suspending theoretical judgments about the external world. This technique of phenomenological reduction is known as "bracketing" or epoché. The goal is to give an unbiased description of the appearance of things. Methodological naturalism places great emphasis on the empirical approach and the resulting theories found in the natural sciences. In this way, it contrasts with methodologies that give more weight to pure reasoning and introspection. 

Relation to other fields 

Philosophy is closely related to many other fields. It is sometimes understood as a meta-discipline that clarifies the nature and limits of other disciplines. It does this by critically examining their basic concepts, background assumptions, and methods. In this regard, it plays a key role in providing an interdisciplinary perspective. It bridges the gap between different disciplines by analyzing which concepts and problems they have in common. It shows how they overlap while also delimiting their scope. Historically, most of the individual sciences originated from philosophy. The influence of philosophy is felt in several fields that require difficult practical decisions. In medicine, philosophical considerations related to bioethics affect issues like whether an embryo is already a person and under what conditions abortion is morally permissible. A closely related philosophical problem is how humans should treat other animals, for instance, whether it is acceptable to use non-human animals as food or for research experiments. In relation to business and professional life, philosophy has contributed by providing ethical frameworks. They contain guidelines on which business practices are morally acceptable and cover the issue of corporate social responsibility. Philosophical inquiry is relevant to many fields that are concerned with what to believe and how to arrive at evidence for one's beliefs. This is a key issue for the sciences, which have as one of their prime objectives the creation of scientific knowledge. Scientific knowledge is based on empirical evidence but it is often not clear whether empirical observations are neutral or already include theoretical assumptions. A closely connected problem is whether the available evidence is sufficient to decide between competing theories. Epistemological problems in relation to the law include what counts as evidence and how much evidence is required to find a person guilty of a crime. A related issue in journalism is how to ensure truth and objectivity when reporting on events. In the fields of theology and religion, there are many doctrines associated with the existence and nature of God as well as rules governing correct behavior. A key issue is whether a rational person should believe these doctrines, for example, whether revelation in the form of holy books and religious experiences of the divine are sufficient evidence for these beliefs. Philosophy in the form of logic has been influential in the fields of mathematics and computer science. Further fields influenced by philosophy include psychology, sociology, linguistics, education, and the arts. The close relation between philosophy and other fields in the contemporary period is reflected in the fact that many philosophy graduates go on to work in related fields rather than in philosophy itself. In the field of politics, philosophy addresses issues such as how to assess whether a government policy is just. Philosophical ideas have prepared and shaped various political developments. For example, ideals formulated in Enlightenment philosophy laid the foundation for constitutional democracy and played a role in the American Revolution and the French Revolution. Marxist philosophy and its exposition of communism was one of the factors in the Russian Revolution and the Chinese Communist Revolution. In India, Mahatma Gandhi's philosophy of non-violence shaped the Indian independence movement. An example of the cultural and critical role of philosophy is found in its influence on the feminist movement through philosophers such as Mary Wollstonecraft, Simone de Beauvoir, and Judith Butler. It has shaped the understanding of key concepts in feminism, for instance, the meaning of gender, how it differs from biological sex, and what role it plays in the formation of personal identity. Philosophers have also investigated the concepts of justice and equality and their implications with respect to the prejudicial treatment of women in male-dominated societies. The idea that philosophy is useful for many aspects of life and society is sometimes rejected. According to some academics, philosophy is mainly undertaken for its own sake and does not make significant contributions to existing practices or external goals. 

See also 

References 

Notes 

Citations 

Bibliography 

External links 

Internet Encyclopedia of Philosophy – a peer-reviewed online encyclopedia of philosophy Stanford Encyclopedia of Philosophy – an online encyclopedia of philosophy maintained by Stanford University PhilPapers – a comprehensive directory of online philosophical articles and books by academic philosophers Internet Philosophy Ontology Project – a model of relationships between philosophical ideas, thinkers, and journals 

 

 
"""
# ───────────────────────────────────────────────────────────── terminal colors

_CODES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
    "red": "\033[31m",
}
_RESET = "\033[0m"
_MASK_STYLED = "\033[41m\033[37m[MASK]\033[0m"


class Term:
    """ANSI painter that auto-disables when not a tty or NO_COLOR is set."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    @classmethod
    def detect(cls, no_color: bool = False, stream=None) -> "Term":
        if stream is None:
            stream = sys.stdout
        is_tty = hasattr(stream, "isatty") and stream.isatty()
        enabled = not no_color and is_tty and os.environ.get("NO_COLOR") is None
        return cls(enabled=enabled)

    def paint(self, color: str, text: str) -> str:
        if not self.enabled:
            return text
        code = _CODES.get(color, "")
        return f"{code}{text}{_RESET}" if code else text

    @property
    def mask(self) -> str:
        return _MASK_STYLED if self.enabled else "[MASK]"


# ───────────────────────────────────────────────────────────── topology

class BidirectionalTopology:
    """Bidirectional n-gram statistics over a corpus.

    For every observed word we count which words appear to its left/right
    at distances 1..max_n. Totals are cached to avoid re-summing.
    Autocomplete uses left side only; right side kept for compatibility
    and for optional diffusion-style generation.
    """

    def __init__(self, max_n: int = 3) -> None:
        if max_n < 1:
            raise ValueError(f"max_n must be >= 1, got {max_n}")
        self.max_n = max_n
        self.left_counts: dict[int, dict[tuple[str, ...], Counter[str]]] = {
            n: defaultdict(Counter) for n in range(1, max_n + 1)
        }
        self.right_counts: dict[int, dict[tuple[str, ...], Counter[str]]] = {
            n: defaultdict(Counter) for n in range(1, max_n + 1)
        }
        self.left_totals: dict[int, dict[tuple[str, ...], int]] = {
            n: {} for n in range(1, max_n + 1)
        }
        self.right_totals: dict[int, dict[tuple[str, ...], int]] = {
            n: {} for n in range(1, max_n + 1)
        }
        self.unigrams: Counter[str] = Counter()
        self.vocab: set[str] = set()
        self.sentences: int = 0
        self.tokens: int = 0

    @classmethod
    def from_text(cls, text: str, max_n: int = 3) -> "BidirectionalTopology":
        topo = cls(max_n=max_n)
        topo.ingest(text)
        return topo

    def ingest(self, text: str) -> None:
        if not text or not text.strip():
            raise ValueError("corpus text is empty")
        flattened = WHITESPACE_RE.sub(" ", text).strip()
        ingested = 0
        for sentence in SENTENCE_SPLIT_RE.split(flattened):
            words = tokenize(sentence)
            if not words:
                continue
            self.sentences += 1
            ingested += len(words)
            self.vocab.update(words)
            self.unigrams.update(words)
            for n in range(1, self.max_n + 1):
                for i, target in enumerate(words):
                    if i >= n:
                        ctx = tuple(words[i - n: i])
                        self.left_counts[n][ctx][target] += 1
                        self.left_totals[n][ctx] = self.left_totals[n].get(ctx, 0) + 1
                    if i + n < len(words):
                        ctx = tuple(words[i + 1: i + 1 + n])
                        self.right_counts[n][ctx][target] += 1
                        self.right_totals[n][ctx] = self.right_totals[n].get(ctx, 0) + 1
        self.tokens += ingested
        if ingested == 0:
            raise ValueError("corpus text contains no usable tokens")


# ───────────────────────────────────────────────────────────── diffusion engine (legacy) + autocomplete

_FLOOR = 1e-5
_UNIGRAM_WEIGHT = 0.1
_BACKOFF_SIZE = 50

StepCallback = Callable[[list[str], int, int], None]

@dataclass
class DenoiseResult:
    sequence: list[str]
    confidences: list[float]

@dataclass
class CompleteResult:
    prefix: list[str]
    continuation: list[str]
    confidences: list[float]
    full_sequence: list[str]


class DiscreteDiffusionEngine:
    """N-gram engine: diffusion (bidirectional, legacy) + causal autocomplete."""

    def __init__(self, topo: BidirectionalTopology, rng: random.Random | None = None) -> None:
        self.topo = topo
        self.rng = rng if rng is not None else random.Random()
        self._backoff = [w for w, _ in topo.unigrams.most_common(_BACKOFF_SIZE)]

    def _active_contexts(self, seq: Sequence[str], idx: int):
        for n in range(1, self.topo.max_n + 1):
            if idx >= n:
                ctx = tuple(seq[idx - n: idx])
                if MASK not in ctx:
                    yield "left", n, ctx
            if idx + n < len(seq):
                ctx = tuple(seq[idx + 1: idx + 1 + n])
                if MASK not in ctx:
                    yield "right", n, ctx

    def _causal_contexts(self, seq: Sequence[str], idx: int):
        """Only left contexts — for autocomplete."""
        for n in range(1, self.topo.max_n + 1):
            if idx >= n:
                ctx = tuple(seq[idx - n: idx])
                if MASK not in ctx:
                    yield "left", n, ctx

    def candidate_distribution(self, seq: Sequence[str], idx: int) -> dict[str, float]:
        """Pure (no mutation) softmax over candidates for position *idx* — bidirectional (legacy)."""
        base = 0.0
        contrib: dict[str, float] = {}
        for side, n, ctx in self._active_contexts(seq, idx):
            if side == "left":
                counts = self.topo.left_counts[n].get(ctx)
                total = self.topo.left_totals[n].get(ctx, 0)
            else:
                counts = self.topo.right_counts[n].get(ctx)
                total = self.topo.right_totals[n].get(ctx, 0)
            if not counts or total <= 0:
                continue
            base += math.log(_FLOOR) * n
            floor_adj = math.log(_FLOOR) * n
            for word, count in counts.items():
                contrib[word] = contrib.get(word, 0.0) + (
                    math.log(count / total + _FLOOR) * n - floor_adj
                )
        candidates = sorted(set(contrib) | set(self._backoff))
        energies = {
            w: math.log(self.topo.unigrams.get(w, 1) + 1) * _UNIGRAM_WEIGHT + base + contrib.get(w, 0.0)
            for w in candidates
        }
        max_e = max(energies.values())
        exps = {w: math.exp(e - max_e) for w, e in energies.items()}
        total = sum(exps.values())
        return {w: e / total for w, e in exps.items()}

    def causal_distribution(self, seq: Sequence[str], idx: int) -> dict[str, float]:
        """Causal (left-only) distribution — used for autocomplete."""
        base = 0.0
        contrib: dict[str, float] = {}
        for _, n, ctx in self._causal_contexts(seq, idx):
            counts = self.topo.left_counts[n].get(ctx)
            total = self.topo.left_totals[n].get(ctx, 0)
            if not counts or total <= 0:
                continue
            base += math.log(_FLOOR) * n
            floor_adj = math.log(_FLOOR) * n
            for word, count in counts.items():
                contrib[word] = contrib.get(word, 0.0) + (
                    math.log(count / total + _FLOOR) * n - floor_adj
                )
        candidates = sorted(set(contrib) | set(self._backoff))
        energies = {
            w: math.log(self.topo.unigrams.get(w, 1) + 1) * _UNIGRAM_WEIGHT + base + contrib.get(w, 0.0)
            for w in candidates
        }
        max_e = max(energies.values())
        exps = {w: math.exp(e - max_e) for w, e in energies.items()}
        total = sum(exps.values())
        return {w: e / total for w, e in exps.items()}

    def complete(
        self,
        prefix: Sequence[str],
        max_tokens: int = 12,
        temperature: float = 0.35,
        threshold: float = 0.0,
    ) -> CompleteResult:
        """Left-to-right autocomplete: continue prefix token by token."""
        seq = [w.lower() for w in prefix]
        confidences: list[float] = []
        continuation: list[str] = []
        for _ in range(max_tokens):
            idx = len(seq)
            probs = self.causal_distribution(seq, idx)
            if not probs:
                break
            # temperature-scaled sampling
            if temperature <= 0:
                chosen = max(probs, key=probs.get)
                conf = probs[chosen]
            else:
                words = list(probs)
                weights = [p ** (1.0 / max(temperature, 1e-6)) for p in probs.values()]
                chosen = self.rng.choices(words, weights=weights, k=1)[0]
                conf = probs[chosen]
            if conf < threshold:
                break
            seq.append(chosen)
            continuation.append(chosen)
            confidences.append(conf)
            # stop early on sentence end if we already generated a few tokens
            if chosen in {".", "!", "?"} and len(continuation) >= 4:
                break
        full_conf = [1.0]*len(prefix) + confidences
        return CompleteResult(prefix=list(prefix), continuation=continuation, confidences=confidences, full_sequence=seq)

    def denoise(
        self,
        target_len: int,
        steps: int,
        prompt: Sequence[str] = (),
        on_step: StepCallback | None = None,
    ) -> DenoiseResult:
        if target_len < 1:
            raise ValueError(f"target_len must be >= 1, got {target_len}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")

        seq = [MASK] * target_len
        locked: set[int] = set()
        for i, w in enumerate(prompt[:target_len]):
            seq[i] = w.lower()
            locked.add(i)

        confidences = [0.0] * target_len
        for t in range(1, steps + 1):
            temp = 1.2 * (1.0 - t / steps) + 0.2
            current: dict[int, float] = {}
            for i in range(target_len):
                if i in locked or seq[i] != MASK:
                    continue
                probs = self.candidate_distribution(seq, i)
                words = list(probs)
                weights = [p ** (1.0 / temp) for p in probs.values()]
                chosen = self.rng.choices(words, weights=weights, k=1)[0]
                seq[i] = chosen
                current[i] = probs[chosen]
                confidences[i] = probs[chosen]
            re_mask_ratio = max(0.0, 1.0 - t / steps)
            num_to_remask = int(len(current) * re_mask_ratio)
            if num_to_remask > 0 and current:
                for idx in sorted(current, key=current.get)[:num_to_remask]:
                    seq[idx] = MASK
                    confidences[idx] = 0.0
            if on_step is not None:
                on_step(list(seq), t, steps)
        return DenoiseResult(sequence=seq, confidences=confidences)


# ───────────────────────────────────────────────────────────── CLI helpers

logger = logging.getLogger("mllm52")
BANNER = r"""MLLM-5.2

"""

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="MLLM-5.2",
        description="Document autocomplete LM — continues your prefix left-to-right using causal n-gram diffusion topology.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python MLLM-5.2.py "the quick brown" --steps 12 --seed 42
              python MLLM-5.2.py autocomplete "what is an atom" --steps 16
              python MLLM-5.2.py generate "hello world" --show-steps --extra-tokens 8 14
              python MLLM-5.2.py --corpus ./my.txt autocomplete "hello"

            tips:
              --steps is the effort/length knob.
              --temperature low (0.2) = deterministic ghost; high (1.0) = creative.
              --seed makes output reproducible.
              In REPL, Tab accepts ghost, Esc dismisses. Try index.html playground!
        """),
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--corpus", type=Path, default=None, help="path to training corpus (default: embedded)")
    p.add_argument("--seed", type=int, default=None, help="seed for reproducible sampling")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    def add_decoding_args(sub):
        sub.add_argument("--steps", type=int, default=16, help="max tokens to generate / diffusion steps (default: 16)")
        sub.add_argument("--extra-tokens", type=int, nargs=2, default=None, metavar=("MIN", "MAX"),
                         help="range of tokens beyond prefix (default: steps..steps, alias for --steps)")
        sub.add_argument("--temperature", type=float, default=0.35, help="sampling temperature 0.0=greedy ..1.2=creative (default 0.35)")
        sub.add_argument("--threshold", type=float, default=0.0, help="min confidence to emit token (default 0.0)")
        sub.add_argument("--max-ngram", type=int, default=3, help="max n-gram order (default 3)")
        sub.add_argument("--show-steps", action="store_true", help="render intermediate diffusion states (generate only)")
        # allow global flags also after subcommand
        sub.add_argument("--seed", type=int, default=None, dest="seed_sub", help=argparse.SUPPRESS)
        sub.add_argument("--corpus", type=Path, default=None, dest="corpus_sub", help=argparse.SUPPRESS)
        sub.add_argument("--no-color", action="store_true", dest="no_color_sub", help=argparse.SUPPRESS)
        sub.add_argument("-v", "--verbose", action="store_true", dest="verbose_sub", help=argparse.SUPPRESS)

    subs = p.add_subparsers(dest="command")
    ac = subs.add_parser("autocomplete", help="autocomplete a prefix (default)")
    ac.add_argument("prompt", nargs="?", default=None, help="prefix text to continue")
    add_decoding_args(ac)

    gen = subs.add_parser("generate", help="alias for autocomplete (legacy diffusion)")
    gen.add_argument("prompt", nargs="?", default=None, help="prefix text")
    add_decoding_args(gen)

    chat = subs.add_parser("chat", help="interactive autocomplete REPL (alias)")
    add_decoding_args(chat)

    comp = subs.add_parser("complete", help=argparse.SUPPRESS)
    comp.add_argument("prompt", nargs="?", default=None, help=argparse.SUPPRESS)
    add_decoding_args(comp)

    return p


def load_topology(corpus_path: Path | None, term: Term, max_n: int = 3) -> BidirectionalTopology:
    t0 = time.time()
    if corpus_path is not None:
        logger.info("loading corpus from %s", corpus_path)
        text = corpus_path.read_text(encoding="utf-8")
        topo = BidirectionalTopology.from_text(text, max_n=max_n)
        src = str(corpus_path)
    else:
        logger.info("using embedded built-in corpus")
        text = BUILT_IN_CORPUS
        topo = BidirectionalTopology.from_text(text, max_n=max_n)
        src = "embedded"
    dt = time.time() - t0
    print(term.paint("dim", f"· corpus: {src}"))
    print(term.paint("dim", f"· topology: {len(topo.vocab)} vocab · {topo.tokens} tokens · {topo.sentences} sentences · {dt:.2f}s  (causal n={max_n})"))
    logger.info("topology: %d vocab %d tokens %d sents in %.2fs", len(topo.vocab), topo.tokens, topo.sentences, dt)
    return topo


def make_step_printer(term: Term):
    def _print(seq: list[str], step: int, total: int) -> None:
        interval = max(1, total // 5)
        if step % interval != 0 and step != total:
            return
        bar_len = 20
        filled = int(bar_len * (step / total))
        bar = "█" * filled + "░" * (bar_len - filled)
        display = "".join(term.paint("red", "_") if w == MASK else term.paint("green", w[0]) for w in seq)
        print(term.paint("dim", f"[Step {step:02d}/{total}] {bar}") + " " + display)
    return _print


def assemble_autocomplete(prefix_words: Sequence[str], cont: Sequence[str]) -> tuple[str, str]:
    prefix_str = " ".join(prefix_words)
    cont_str = " ".join(cont)
    for mark in (",", ".", "!", "?"):
        cont_str = cont_str.replace(f" {mark}", mark)
    if prefix_str and cont_str and not prefix_str.endswith((" ", ".", "!", "?")):
        prefix_str += " "
    if cont_str:
        # capitalize first char of continuation if prefix ends with sentence boundary
        if not prefix_str or prefix_str.rstrip().endswith((".", "!", "?")):
            cont_str = cont_str[0].upper() + cont_str[1:] if cont_str else cont_str
    return prefix_str, cont_str


def assemble_text(prompt_words: Sequence[str], result: DenoiseResult) -> tuple[str, str]:
    prompt_str = " ".join(prompt_words)
    gen_words = [w for i, w in enumerate(result.sequence) if i >= len(prompt_words) and w != MASK]
    gen_str = " ".join(gen_words)
    for mark in (",", ".", "!", "?"):
        gen_str = gen_str.replace(f" {mark}", mark)
    if prompt_str and not prompt_str.endswith((" ", ".", "!", "?")):
        prompt_str += " "
    if gen_str:
        gen_str = gen_str[0].upper() + gen_str[1:]
    return prompt_str, gen_str


def render_autocomplete(term: Term, prefix_words: Sequence[str], result: CompleteResult, show_heatmap: bool = True) -> None:
    prefix_str, cont_str = assemble_autocomplete(prefix_words, result.continuation)
    print()
    print(term.paint("cyan", term.paint("bold", "Autocomplete:")))
    # ghost style: prefix normal, continuation dim/italic-like
    print(term.paint("cyan", prefix_str) + term.paint("dim", cont_str))
    if show_heatmap and result.continuation:
        print()
        print(term.paint("dim", "Confidence:"))
        chips = []
        for c in result.confidences:
            color = "green" if c > 0.35 else "yellow" if c > 0.15 else "red"
            chips.append(term.paint(color, "█"))
        print("".join(chips) + term.paint("dim", f"  avg {sum(result.confidences)/len(result.confidences):.2f}"))
        print(term.paint("dim", "[Green=High  Yellow=Med  Red=Low]  Tab=accept  Esc=dismiss"))


def render_result(term: Term, prompt_words: Sequence[str], result: DenoiseResult) -> None:
    prompt_str, gen_str = assemble_text(prompt_words, result)
    print()
    print(term.paint("cyan", term.paint("bold", "Full Sculpted Text:")))
    print(term.paint("cyan", prompt_str) + term.paint("magenta", term.paint("bold", gen_str)))
    heat = []
    for i, word in enumerate(result.sequence):
        if word == MASK:
            continue
        if i < len(prompt_words):
            heat.append(term.paint("cyan", "█"))
        else:
            c = result.confidences[i]
            color = "green" if c > 0.6 else "yellow" if c > 0.3 else "red"
            heat.append(term.paint(color, "█"))
    print()
    print(term.paint("dim", "Confidence Heatmap (Prompt | Generated):"))
    print("".join(heat))
    print(term.paint("dim", "[Green=High  Yellow=Med  Red=Low]"))


def _get_max_tokens(args) -> int:
    if getattr(args, "extra_tokens", None):
        lo, hi = args.extra_tokens
        # for autocomplete, use hi as max
        return int(hi)
    return int(getattr(args, "steps", 16))


def do_autocomplete(engine: DiscreteDiffusionEngine, text: str, max_tokens: int, temperature: float, threshold: float) -> tuple[list[str], CompleteResult]:
    prefix_words = tokenize(text) if text else []
    res = engine.complete(prefix=prefix_words, max_tokens=max_tokens, temperature=temperature, threshold=threshold)
    return prefix_words, res


def cmd_autocomplete(args: argparse.Namespace, engine: DiscreteDiffusionEngine, term: Term) -> int:
    # prompt may be in args.prompt or global prefix
    prompt_text = getattr(args, "prompt", None) or getattr(args, "prefix", None) or ""
    if not prompt_text and not sys.stdin.isatty():
        try:
            _stdin = sys.stdin.read().strip()
        except Exception:
            _stdin = ""
        if _stdin:
            prompt_text = _stdin
    if not prompt_text:
        # if no prompt provided via CLI, fall back to REPL
        return cmd_chat(args, engine, term)
    max_tokens = _get_max_tokens(args)
    prefix_words, result = do_autocomplete(engine, prompt_text, max_tokens, float(getattr(args, "temperature", 0.35)), float(getattr(args, "threshold", 0.0)))
    render_autocomplete(term, prefix_words, result)
    return 0


def cmd_generate(args: argparse.Namespace, engine: DiscreteDiffusionEngine, term: Term) -> int:
    # legacy diffusion path if --show-steps requested, else use autocomplete
    if getattr(args, "show_steps", False):
        on_step = make_step_printer(term)
        prompt_text = getattr(args, "prompt", None) or getattr(args, "prefix", None) or ""
        if not prompt_text and not sys.stdin.isatty():
            try:
                _stdin = sys.stdin.read().strip()
            except Exception:
                _stdin = ""
            if _stdin:
                prompt_text = _stdin
        prompt_words = tokenize(prompt_text) if prompt_text else []
        lo, hi = tuple(args.extra_tokens) if args.extra_tokens else (int(args.steps), int(args.steps))
        # for legacy we keep diffusion: use hi as extra
        target_len = len(prompt_words) + engine.rng.randint(lo, hi)
        result = engine.denoise(target_len=target_len, steps=int(args.steps), prompt=prompt_words, on_step=on_step)
        render_result(term, prompt_words, result)
        return 0
    # otherwise autocomplete
    return cmd_autocomplete(args, engine, term)


def cmd_chat(args: argparse.Namespace, engine: DiscreteDiffusionEngine, term: Term) -> int:
    print(term.paint("magenta", term.paint("bold", "\n› MLLM-5.2 Autocomplete — type a prefix, Tab to accept ghost.  :help  [quit] to exit.")))
    print(term.paint("dim", "  (causal n-gram, left context only — continues your document)\n"))
    while True:
        try:
            raw = input(term.paint("bold", "› ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        low = raw.lower()
        if low in {"[quit]", "quit", "exit", ":quit", ":q"}:
            break
        if low in {":help", "help", "?"}:
            print(term.paint("dim", "  commands: [quit]/exit  :help  :clear  :steps N  :temp N  :seed N  :show on|off"))
            print(term.paint("dim", "  flags: --steps N --temperature T --threshold T --max-ngram N --seed N --corpus PATH"))
            print(term.paint("dim", "  playground: open index.html in browser for ghost-text editor"))
            continue
        if low in {":clear", "clear"}:
            os.system("clear" if os.name != "nt" else "cls")
            continue
        if low.startswith(":steps"):
            try:
                n = int(low.split()[1])
                if n < 1: raise ValueError
                args.steps = n
                print(term.paint("dim", f"  steps → {n}"))
            except Exception:
                print(term.paint("red", "  usage: :steps <positive int>"))
            continue
        if low.startswith(":temp"):
            try:
                t = float(low.split()[1])
                args.temperature = t
                print(term.paint("dim", f"  temperature → {t}"))
            except Exception:
                print(term.paint("red", "  usage: :temp <float 0.0-1.2>"))
            continue
        if low.startswith(":seed"):
            try:
                if low.split()[1].lower() == "none":
                    engine.rng = random.Random()
                    print(term.paint("dim", "  seed → random"))
                else:
                    s = int(low.split()[1])
                    engine.rng = random.Random(s)
                    print(term.paint("dim", f"  seed → {s}"))
            except Exception:
                print(term.paint("red", "  usage: :seed <int>  or  :seed none"))
            continue

        # treat input as prefix to autocomplete
        max_tokens = _get_max_tokens(args)
        prefix_words, result = do_autocomplete(engine, raw, max_tokens, float(getattr(args, "temperature", 0.35)), float(getattr(args, "threshold", 0.0)))
        render_autocomplete(term, prefix_words, result)
        print()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    # ── Bare-prefix compatibility: support `python MLLM-5.2.py "hello world" --steps 4`
    # without requiring explicit subcommand, like ghost-text one-shot.
    # This keeps 5.1 architecture (BidirectionalTopology, DiscreteDiffusionEngine, etc.)
    # intact while making one-shot intuitive. If no known subcommand is present,
    # parse flags loosely and treat remaining as prompt (or stdin pipe).
    if argv is None:
        _argv_list = sys.argv[1:]
    else:
        _argv_list = list(argv)
    known_subs = {"autocomplete", "generate", "chat", "complete"}
    has_sub = any(tok in known_subs for tok in _argv_list)
    has_help = any(tok in ("-h", "--help", "--version") for tok in _argv_list)
    if not has_sub and not has_help:
        _tmp = argparse.ArgumentParser(add_help=False)
        _tmp.add_argument("--corpus", type=Path, default=None)
        _tmp.add_argument("--seed", type=int, default=None)
        _tmp.add_argument("--no-color", action="store_true")
        _tmp.add_argument("-v", "--verbose", action="store_true")
        _tmp.add_argument("--steps", type=int, default=16)
        _tmp.add_argument("--extra-tokens", type=int, nargs=2, default=None, metavar=("MIN", "MAX"))
        _tmp.add_argument("--temperature", type=float, default=0.35)
        _tmp.add_argument("--threshold", type=float, default=0.0)
        _tmp.add_argument("--max-ngram", type=int, default=3)
        _tmp.add_argument("--show-steps", action="store_true")
        try:
            _tmp_args, _remaining = _tmp.parse_known_args(_argv_list)
        except SystemExit:
            _tmp_args = None
            _remaining = None  # type: ignore
        if _tmp_args is not None and _remaining is not None:
            _unknown = [t for t in _remaining if t.startswith("-")]
            if not _unknown:
                _prompt_text = " ".join(_remaining).strip()
                _stdin_text = ""
                if not _prompt_text and not sys.stdin.isatty():
                    try:
                        _stdin_text = sys.stdin.read().strip()
                    except Exception:
                        _stdin_text = ""
                    if _stdin_text:
                        _prompt_text = _stdin_text
                # Only take bare path if there's a prompt or stdin or no args (REPL)
                # If _remaining is empty and no stdin, this is a bare REPL invocation -> handle here too
                # Distinguish from unknown-flag case which we already excluded
                # Proceed with bare handling (autocomplete or REPL)
                _seed = _tmp_args.seed
                _corpus_arg = _tmp_args.corpus
                _no_color = bool(_tmp_args.no_color)
                _verbose = bool(_tmp_args.verbose)
                logging.basicConfig(level=logging.DEBUG if _verbose else logging.WARNING,
                                    format="%(levelname)s %(name)s: %(message)s")
                _term = Term.detect(_no_color)
                if _term.enabled:
                    print(_term.paint("cyan", BANNER.strip("\n")))
                    print(_term.paint("dim", f"  v{__version__}  ·  autocomplete LM  ·  causal n={_tmp_args.max_ngram}  ·  ghost-text"))
                    print()
                else:
                    print(f"MLLM-5.2 v{__version__} — autocomplete LM")
                _extra = tuple(_tmp_args.extra_tokens) if _tmp_args.extra_tokens else None
                if _extra and len(_extra) == 2 and _extra[0] > _extra[1]:
                    print("mllm52: error: --extra-tokens MIN must be <= MAX", file=sys.stderr)
                    return 2
                _max_n = int(_tmp_args.max_ngram)
                _corpus_path = Path(_corpus_arg) if _corpus_arg else None
                try:
                    _topo = load_topology(_corpus_path, _term, max_n=_max_n)
                except (OSError, ValueError) as exc:
                    print(f"mllm52: error: cannot load corpus {_corpus_path or 'embedded'}: {exc}", file=sys.stderr)
                    return 2
                if _extra:
                    _tmp_args.extra_tokens = _extra
                else:
                    _tmp_args.extra_tokens = (int(_tmp_args.steps), int(_tmp_args.steps))
                _engine = DiscreteDiffusionEngine(_topo, rng=random.Random(_seed))
                _pseudo = argparse.Namespace(
                    prompt=_prompt_text if _prompt_text else None,
                    prefix=None,
                    steps=_tmp_args.steps,
                    extra_tokens=_tmp_args.extra_tokens,
                    temperature=_tmp_args.temperature,
                    threshold=_tmp_args.threshold,
                    max_ngram=_tmp_args.max_ngram,
                    show_steps=_tmp_args.show_steps,
                    seed=_seed,
                    corpus=_corpus_arg,
                    no_color=_no_color,
                    verbose=_verbose,
                    command="autocomplete" if _prompt_text else None,
                )
                if _prompt_text:
                    if _tmp_args.show_steps:
                        return cmd_generate(_pseudo, _engine, _term)
                    return cmd_autocomplete(_pseudo, _engine, _term)
                else:
                    # No prompt and no stdin -> REPL, but only if not already handled as help
                    # If argv_list was empty, this is plain REPL invocation
                    # Also if argv_list contained only flags, go to REPL with those flags
                    return cmd_chat(_pseudo, _engine, _term)
    parser = build_parser()
    args = parser.parse_args(argv)
    # allow global flags also after subcommand
    seed = getattr(args, "seed_sub", None) if getattr(args, "seed_sub", None) is not None else getattr(args, "seed", None)
    corpus_arg = getattr(args, "corpus_sub", None) if getattr(args, "corpus_sub", None) is not None else getattr(args, "corpus", None)
    no_color = bool(getattr(args, "no_color", False) or getattr(args, "no_color_sub", False))
    verbose = bool(getattr(args, "verbose", False) or getattr(args, "verbose_sub", False))

    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    term = Term.detect(no_color)

    if term.enabled:
        print(term.paint("cyan", BANNER.strip("\n")))
        print(term.paint("dim", f"  v{__version__}  ·  autocomplete LM  ·  causal n=3  ·  ghost-text"))
        print()
    else:
        print(f"MLLM-5.2 v{__version__} — autocomplete LM")



    # validate extra-tokens
    try:
        extra = tuple(args.extra_tokens) if getattr(args, "extra_tokens", None) else None
    except Exception:
        extra = None
    if extra and len(extra) == 2 and extra[0] > extra[1]:
        print("mllm52: error: --extra-tokens MIN must be <= MAX", file=sys.stderr)
        return 2

    max_n = int(getattr(args, "max_ngram", 3)) if hasattr(args, "max_ngram") else 3
    corpus_path = Path(corpus_arg) if corpus_arg else None
    try:
        topo = load_topology(corpus_path, term, max_n=max_n)
    except (OSError, ValueError) as exc:
        print(f"mllm52: error: cannot load corpus {corpus_path or 'embedded'}: {exc}", file=sys.stderr)
        return 2

    # patch args for downstream
    if extra:
        args.extra_tokens = extra
    else:
        # default extra_tokens = steps..steps for autocomplete
        args.extra_tokens = (int(getattr(args, "steps", 16)), int(getattr(args, "steps", 16)))
    if not hasattr(args, "steps") or args.steps is None:
        args.steps = 16
    if not hasattr(args, "temperature"):
        args.temperature = 0.35
    if not hasattr(args, "threshold"):
        args.threshold = 0.0

    engine = DiscreteDiffusionEngine(topo, rng=random.Random(seed))
    cmd = getattr(args, "command", None)
    if cmd in ("autocomplete", "complete"):
        # Let cmd_autocomplete handle prompt or stdin or fallback to REPL
        return cmd_autocomplete(args, engine, term)
    if cmd is None and getattr(args, "prefix", None) is None:
        # No subcommand and no prefix attribute -> default REPL (or bare already handled)
        # Check stdin for one-shot without subcommand (should have been handled earlier, but keep)
        if not sys.stdin.isatty():
            try:
                _stdin_prompt = sys.stdin.read().strip()
            except Exception:
                _stdin_prompt = ""
            if _stdin_prompt:
                _pseudo = argparse.Namespace(prompt=_stdin_prompt, prefix=None, steps=getattr(args, "steps", 16),
                                             extra_tokens=getattr(args, "extra_tokens", None),
                                             temperature=getattr(args, "temperature", 0.35),
                                             threshold=getattr(args, "threshold", 0.0),
                                             max_ngram=getattr(args, "max_ngram", 3),
                                             show_steps=getattr(args, "show_steps", False))
                return cmd_autocomplete(_pseudo, engine, term)
        return cmd_chat(args, engine, term)
    if cmd == "generate":
        return cmd_generate(args, engine, term)
    if cmd == "chat":
        return cmd_chat(args, engine, term)
    return cmd_chat(args, engine, term)


if __name__ == "__main__":
    raise SystemExit(main())

 
"""
# ───────────────────────────────────────────────────────────── terminal colors

_CODES = {
    "bold": "\033[1m",
    "dim": "\033[2m",
    "cyan": "\033[36m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "magenta": "\033[35m",
    "red": "\033[31m",
}
_RESET = "\033[0m"
_MASK_STYLED = "\033[41m\033[37m[MASK]\033[0m"


class Term:
    """ANSI painter that auto-disables when not a tty or NO_COLOR is set."""

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    @classmethod
    def detect(cls, no_color: bool = False, stream=None) -> "Term":
        if stream is None:
            stream = sys.stdout
        is_tty = hasattr(stream, "isatty") and stream.isatty()
        enabled = not no_color and is_tty and os.environ.get("NO_COLOR") is None
        return cls(enabled=enabled)

    def paint(self, color: str, text: str) -> str:
        if not self.enabled:
            return text
        code = _CODES.get(color, "")
        return f"{code}{text}{_RESET}" if code else text

    @property
    def mask(self) -> str:
        return _MASK_STYLED if self.enabled else "[MASK]"


# ───────────────────────────────────────────────────────────── topology

class BidirectionalTopology:
    """Bidirectional n-gram statistics over a corpus.

    For every observed word we count which words appear to its left/right
    at distances 1..max_n. Totals are cached to avoid re-summing.
    Autocomplete uses left side only; right side kept for compatibility
    and for optional diffusion-style generation.
    """

    def __init__(self, max_n: int = 3) -> None:
        if max_n < 1:
            raise ValueError(f"max_n must be >= 1, got {max_n}")
        self.max_n = max_n
        self.left_counts: dict[int, dict[tuple[str, ...], Counter[str]]] = {
            n: defaultdict(Counter) for n in range(1, max_n + 1)
        }
        self.right_counts: dict[int, dict[tuple[str, ...], Counter[str]]] = {
            n: defaultdict(Counter) for n in range(1, max_n + 1)
        }
        self.left_totals: dict[int, dict[tuple[str, ...], int]] = {
            n: {} for n in range(1, max_n + 1)
        }
        self.right_totals: dict[int, dict[tuple[str, ...], int]] = {
            n: {} for n in range(1, max_n + 1)
        }
        self.unigrams: Counter[str] = Counter()
        self.vocab: set[str] = set()
        self.sentences: int = 0
        self.tokens: int = 0

    @classmethod
    def from_text(cls, text: str, max_n: int = 3) -> "BidirectionalTopology":
        topo = cls(max_n=max_n)
        topo.ingest(text)
        return topo

    def ingest(self, text: str) -> None:
        if not text or not text.strip():
            raise ValueError("corpus text is empty")
        flattened = WHITESPACE_RE.sub(" ", text).strip()
        ingested = 0
        for sentence in SENTENCE_SPLIT_RE.split(flattened):
            words = tokenize(sentence)
            if not words:
                continue
            self.sentences += 1
            ingested += len(words)
            self.vocab.update(words)
            self.unigrams.update(words)
            for n in range(1, self.max_n + 1):
                for i, target in enumerate(words):
                    if i >= n:
                        ctx = tuple(words[i - n: i])
                        self.left_counts[n][ctx][target] += 1
                        self.left_totals[n][ctx] = self.left_totals[n].get(ctx, 0) + 1
                    if i + n < len(words):
                        ctx = tuple(words[i + 1: i + 1 + n])
                        self.right_counts[n][ctx][target] += 1
                        self.right_totals[n][ctx] = self.right_totals[n].get(ctx, 0) + 1
        self.tokens += ingested
        if ingested == 0:
            raise ValueError("corpus text contains no usable tokens")


# ───────────────────────────────────────────────────────────── diffusion engine (legacy) + autocomplete

_FLOOR = 1e-5
_UNIGRAM_WEIGHT = 0.1
_BACKOFF_SIZE = 50

StepCallback = Callable[[list[str], int, int], None]

@dataclass
class DenoiseResult:
    sequence: list[str]
    confidences: list[float]

@dataclass
class CompleteResult:
    prefix: list[str]
    continuation: list[str]
    confidences: list[float]
    full_sequence: list[str]


class DiscreteDiffusionEngine:
    """N-gram engine: diffusion (bidirectional, legacy) + causal autocomplete."""

    def __init__(self, topo: BidirectionalTopology, rng: random.Random | None = None) -> None:
        self.topo = topo
        self.rng = rng if rng is not None else random.Random()
        self._backoff = [w for w, _ in topo.unigrams.most_common(_BACKOFF_SIZE)]

    def _active_contexts(self, seq: Sequence[str], idx: int):
        for n in range(1, self.topo.max_n + 1):
            if idx >= n:
                ctx = tuple(seq[idx - n: idx])
                if MASK not in ctx:
                    yield "left", n, ctx
            if idx + n < len(seq):
                ctx = tuple(seq[idx + 1: idx + 1 + n])
                if MASK not in ctx:
                    yield "right", n, ctx

    def _causal_contexts(self, seq: Sequence[str], idx: int):
        """Only left contexts — for autocomplete."""
        for n in range(1, self.topo.max_n + 1):
            if idx >= n:
                ctx = tuple(seq[idx - n: idx])
                if MASK not in ctx:
                    yield "left", n, ctx

    def candidate_distribution(self, seq: Sequence[str], idx: int) -> dict[str, float]:
        """Pure (no mutation) softmax over candidates for position *idx* — bidirectional (legacy)."""
        base = 0.0
        contrib: dict[str, float] = {}
        for side, n, ctx in self._active_contexts(seq, idx):
            if side == "left":
                counts = self.topo.left_counts[n].get(ctx)
                total = self.topo.left_totals[n].get(ctx, 0)
            else:
                counts = self.topo.right_counts[n].get(ctx)
                total = self.topo.right_totals[n].get(ctx, 0)
            if not counts or total <= 0:
                continue
            base += math.log(_FLOOR) * n
            floor_adj = math.log(_FLOOR) * n
            for word, count in counts.items():
                contrib[word] = contrib.get(word, 0.0) + (
                    math.log(count / total + _FLOOR) * n - floor_adj
                )
        candidates = sorted(set(contrib) | set(self._backoff))
        energies = {
            w: math.log(self.topo.unigrams.get(w, 1) + 1) * _UNIGRAM_WEIGHT + base + contrib.get(w, 0.0)
            for w in candidates
        }
        max_e = max(energies.values())
        exps = {w: math.exp(e - max_e) for w, e in energies.items()}
        total = sum(exps.values())
        return {w: e / total for w, e in exps.items()}

    def causal_distribution(self, seq: Sequence[str], idx: int) -> dict[str, float]:
        """Causal (left-only) distribution — used for autocomplete."""
        base = 0.0
        contrib: dict[str, float] = {}
        for _, n, ctx in self._causal_contexts(seq, idx):
            counts = self.topo.left_counts[n].get(ctx)
            total = self.topo.left_totals[n].get(ctx, 0)
            if not counts or total <= 0:
                continue
            base += math.log(_FLOOR) * n
            floor_adj = math.log(_FLOOR) * n
            for word, count in counts.items():
                contrib[word] = contrib.get(word, 0.0) + (
                    math.log(count / total + _FLOOR) * n - floor_adj
                )
        candidates = sorted(set(contrib) | set(self._backoff))
        energies = {
            w: math.log(self.topo.unigrams.get(w, 1) + 1) * _UNIGRAM_WEIGHT + base + contrib.get(w, 0.0)
            for w in candidates
        }
        max_e = max(energies.values())
        exps = {w: math.exp(e - max_e) for w, e in energies.items()}
        total = sum(exps.values())
        return {w: e / total for w, e in exps.items()}

    def complete(
        self,
        prefix: Sequence[str],
        max_tokens: int = 12,
        temperature: float = 0.35,
        threshold: float = 0.0,
    ) -> CompleteResult:
        """Left-to-right autocomplete: continue prefix token by token."""
        seq = [w.lower() for w in prefix]
        confidences: list[float] = []
        continuation: list[str] = []
        for _ in range(max_tokens):
            idx = len(seq)
            probs = self.causal_distribution(seq, idx)
            if not probs:
                break
            # temperature-scaled sampling
            if temperature <= 0:
                chosen = max(probs, key=probs.get)
                conf = probs[chosen]
            else:
                words = list(probs)
                weights = [p ** (1.0 / max(temperature, 1e-6)) for p in probs.values()]
                chosen = self.rng.choices(words, weights=weights, k=1)[0]
                conf = probs[chosen]
            if conf < threshold:
                break
            seq.append(chosen)
            continuation.append(chosen)
            confidences.append(conf)
            # stop early on sentence end if we already generated a few tokens
            if chosen in {".", "!", "?"} and len(continuation) >= 4:
                break
        full_conf = [1.0]*len(prefix) + confidences
        return CompleteResult(prefix=list(prefix), continuation=continuation, confidences=confidences, full_sequence=seq)

    def denoise(
        self,
        target_len: int,
        steps: int,
        prompt: Sequence[str] = (),
        on_step: StepCallback | None = None,
    ) -> DenoiseResult:
        if target_len < 1:
            raise ValueError(f"target_len must be >= 1, got {target_len}")
        if steps < 1:
            raise ValueError(f"steps must be >= 1, got {steps}")

        seq = [MASK] * target_len
        locked: set[int] = set()
        for i, w in enumerate(prompt[:target_len]):
            seq[i] = w.lower()
            locked.add(i)

        confidences = [0.0] * target_len
        for t in range(1, steps + 1):
            temp = 1.2 * (1.0 - t / steps) + 0.2
            current: dict[int, float] = {}
            for i in range(target_len):
                if i in locked or seq[i] != MASK:
                    continue
                probs = self.candidate_distribution(seq, i)
                words = list(probs)
                weights = [p ** (1.0 / temp) for p in probs.values()]
                chosen = self.rng.choices(words, weights=weights, k=1)[0]
                seq[i] = chosen
                current[i] = probs[chosen]
                confidences[i] = probs[chosen]
            re_mask_ratio = max(0.0, 1.0 - t / steps)
            num_to_remask = int(len(current) * re_mask_ratio)
            if num_to_remask > 0 and current:
                for idx in sorted(current, key=current.get)[:num_to_remask]:
                    seq[idx] = MASK
                    confidences[idx] = 0.0
            if on_step is not None:
                on_step(list(seq), t, steps)
        return DenoiseResult(sequence=seq, confidences=confidences)


# ───────────────────────────────────────────────────────────── CLI helpers

logger = logging.getLogger("mllm52")
BANNER = r"""MLLM-5.2

"""

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="MLLM-5.2",
        description="Document autocomplete LM — continues your prefix left-to-right using causal n-gram diffusion topology.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=textwrap.dedent("""\
            examples:
              python MLLM-5.2.py "the quick brown" --steps 12 --seed 42
              python MLLM-5.2.py autocomplete "what is an atom" --steps 16
              python MLLM-5.2.py generate "hello world" --show-steps --extra-tokens 8 14
              python MLLM-5.2.py --corpus ./my.txt autocomplete "hello"

            tips:
              --steps is the effort/length knob.
              --temperature low (0.2) = deterministic ghost; high (1.0) = creative.
              --seed makes output reproducible.
              In REPL, Tab accepts ghost, Esc dismisses. Try index.html playground!
        """),
    )
    p.add_argument("--version", action="version", version=f"%(prog)s {__version__}")
    p.add_argument("--corpus", type=Path, default=None, help="path to training corpus (default: embedded)")
    p.add_argument("--seed", type=int, default=None, help="seed for reproducible sampling")
    p.add_argument("--no-color", action="store_true", help="disable ANSI colors")
    p.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    def add_decoding_args(sub):
        sub.add_argument("--steps", type=int, default=16, help="max tokens to generate / diffusion steps (default: 16)")
        sub.add_argument("--extra-tokens", type=int, nargs=2, default=None, metavar=("MIN", "MAX"),
                         help="range of tokens beyond prefix (default: steps..steps, alias for --steps)")
        sub.add_argument("--temperature", type=float, default=0.35, help="sampling temperature 0.0=greedy ..1.2=creative (default 0.35)")
        sub.add_argument("--threshold", type=float, default=0.0, help="min confidence to emit token (default 0.0)")
        sub.add_argument("--max-ngram", type=int, default=3, help="max n-gram order (default 3)")
        sub.add_argument("--show-steps", action="store_true", help="render intermediate diffusion states (generate only)")
        # allow global flags also after subcommand
        sub.add_argument("--seed", type=int, default=None, dest="seed_sub", help=argparse.SUPPRESS)
        sub.add_argument("--corpus", type=Path, default=None, dest="corpus_sub", help=argparse.SUPPRESS)
        sub.add_argument("--no-color", action="store_true", dest="no_color_sub", help=argparse.SUPPRESS)
        sub.add_argument("-v", "--verbose", action="store_true", dest="verbose_sub", help=argparse.SUPPRESS)

    subs = p.add_subparsers(dest="command")
    ac = subs.add_parser("autocomplete", help="autocomplete a prefix (default)")
    ac.add_argument("prompt", nargs="?", default=None, help="prefix text to continue")
    add_decoding_args(ac)

    gen = subs.add_parser("generate", help="alias for autocomplete (legacy diffusion)")
    gen.add_argument("prompt", nargs="?", default=None, help="prefix text")
    add_decoding_args(gen)

    chat = subs.add_parser("chat", help="interactive autocomplete REPL (alias)")
    add_decoding_args(chat)

    comp = subs.add_parser("complete", help=argparse.SUPPRESS)
    comp.add_argument("prompt", nargs="?", default=None, help=argparse.SUPPRESS)
    add_decoding_args(comp)

    return p


def load_topology(corpus_path: Path | None, term: Term, max_n: int = 3) -> BidirectionalTopology:
    t0 = time.time()
    if corpus_path is not None:
        logger.info("loading corpus from %s", corpus_path)
        text = corpus_path.read_text(encoding="utf-8")
        topo = BidirectionalTopology.from_text(text, max_n=max_n)
        src = str(corpus_path)
    else:
        logger.info("using embedded built-in corpus")
        text = BUILT_IN_CORPUS
        topo = BidirectionalTopology.from_text(text, max_n=max_n)
        src = "embedded"
    dt = time.time() - t0
    print(term.paint("dim", f"· corpus: {src}"))
    print(term.paint("dim", f"· topology: {len(topo.vocab)} vocab · {topo.tokens} tokens · {topo.sentences} sentences · {dt:.2f}s  (causal n={max_n})"))
    logger.info("topology: %d vocab %d tokens %d sents in %.2fs", len(topo.vocab), topo.tokens, topo.sentences, dt)
    return topo


def make_step_printer(term: Term):
    def _print(seq: list[str], step: int, total: int) -> None:
        interval = max(1, total // 5)
        if step % interval != 0 and step != total:
            return
        bar_len = 20
        filled = int(bar_len * (step / total))
        bar = "█" * filled + "░" * (bar_len - filled)
        display = "".join(term.paint("red", "_") if w == MASK else term.paint("green", w[0]) for w in seq)
        print(term.paint("dim", f"[Step {step:02d}/{total}] {bar}") + " " + display)
    return _print


def assemble_autocomplete(prefix_words: Sequence[str], cont: Sequence[str]) -> tuple[str, str]:
    prefix_str = " ".join(prefix_words)
    cont_str = " ".join(cont)
    for mark in (",", ".", "!", "?"):
        cont_str = cont_str.replace(f" {mark}", mark)
    if prefix_str and cont_str and not prefix_str.endswith((" ", ".", "!", "?")):
        prefix_str += " "
    if cont_str:
        # capitalize first char of continuation if prefix ends with sentence boundary
        if not prefix_str or prefix_str.rstrip().endswith((".", "!", "?")):
            cont_str = cont_str[0].upper() + cont_str[1:] if cont_str else cont_str
    return prefix_str, cont_str


def assemble_text(prompt_words: Sequence[str], result: DenoiseResult) -> tuple[str, str]:
    prompt_str = " ".join(prompt_words)
    gen_words = [w for i, w in enumerate(result.sequence) if i >= len(prompt_words) and w != MASK]
    gen_str = " ".join(gen_words)
    for mark in (",", ".", "!", "?"):
        gen_str = gen_str.replace(f" {mark}", mark)
    if prompt_str and not prompt_str.endswith((" ", ".", "!", "?")):
        prompt_str += " "
    if gen_str:
        gen_str = gen_str[0].upper() + gen_str[1:]
    return prompt_str, gen_str


def render_autocomplete(term: Term, prefix_words: Sequence[str], result: CompleteResult, show_heatmap: bool = True) -> None:
    prefix_str, cont_str = assemble_autocomplete(prefix_words, result.continuation)
    print()
    print(term.paint("cyan", term.paint("bold", "Autocomplete:")))
    # ghost style: prefix normal, continuation dim/italic-like
    print(term.paint("cyan", prefix_str) + term.paint("dim", cont_str))
    if show_heatmap and result.continuation:
        print()
        print(term.paint("dim", "Confidence:"))
        chips = []
        for c in result.confidences:
            color = "green" if c > 0.35 else "yellow" if c > 0.15 else "red"
            chips.append(term.paint(color, "█"))
        print("".join(chips) + term.paint("dim", f"  avg {sum(result.confidences)/len(result.confidences):.2f}"))
        print(term.paint("dim", "[Green=High  Yellow=Med  Red=Low]  Tab=accept  Esc=dismiss"))


def render_result(term: Term, prompt_words: Sequence[str], result: DenoiseResult) -> None:
    prompt_str, gen_str = assemble_text(prompt_words, result)
    print()
    print(term.paint("cyan", term.paint("bold", "Full Sculpted Text:")))
    print(term.paint("cyan", prompt_str) + term.paint("magenta", term.paint("bold", gen_str)))
    heat = []
    for i, word in enumerate(result.sequence):
        if word == MASK:
            continue
        if i < len(prompt_words):
            heat.append(term.paint("cyan", "█"))
        else:
            c = result.confidences[i]
            color = "green" if c > 0.6 else "yellow" if c > 0.3 else "red"
            heat.append(term.paint(color, "█"))
    print()
    print(term.paint("dim", "Confidence Heatmap (Prompt | Generated):"))
    print("".join(heat))
    print(term.paint("dim", "[Green=High  Yellow=Med  Red=Low]"))


def _get_max_tokens(args) -> int:
    if getattr(args, "extra_tokens", None):
        lo, hi = args.extra_tokens
        # for autocomplete, use hi as max
        return int(hi)
    return int(getattr(args, "steps", 16))


def do_autocomplete(engine: DiscreteDiffusionEngine, text: str, max_tokens: int, temperature: float, threshold: float) -> tuple[list[str], CompleteResult]:
    prefix_words = tokenize(text) if text else []
    res = engine.complete(prefix=prefix_words, max_tokens=max_tokens, temperature=temperature, threshold=threshold)
    return prefix_words, res


def cmd_autocomplete(args: argparse.Namespace, engine: DiscreteDiffusionEngine, term: Term) -> int:
    # prompt may be in args.prompt or global prefix
    prompt_text = getattr(args, "prompt", None) or getattr(args, "prefix", None) or ""
    if not prompt_text and not sys.stdin.isatty():
        try:
            _stdin = sys.stdin.read().strip()
        except Exception:
            _stdin = ""
        if _stdin:
            prompt_text = _stdin
    if not prompt_text:
        # if no prompt provided via CLI, fall back to REPL
        return cmd_chat(args, engine, term)
    max_tokens = _get_max_tokens(args)
    prefix_words, result = do_autocomplete(engine, prompt_text, max_tokens, float(getattr(args, "temperature", 0.35)), float(getattr(args, "threshold", 0.0)))
    render_autocomplete(term, prefix_words, result)
    return 0


def cmd_generate(args: argparse.Namespace, engine: DiscreteDiffusionEngine, term: Term) -> int:
    # legacy diffusion path if --show-steps requested, else use autocomplete
    if getattr(args, "show_steps", False):
        on_step = make_step_printer(term)
        prompt_text = getattr(args, "prompt", None) or getattr(args, "prefix", None) or ""
        if not prompt_text and not sys.stdin.isatty():
            try:
                _stdin = sys.stdin.read().strip()
            except Exception:
                _stdin = ""
            if _stdin:
                prompt_text = _stdin
        prompt_words = tokenize(prompt_text) if prompt_text else []
        lo, hi = tuple(args.extra_tokens) if args.extra_tokens else (int(args.steps), int(args.steps))
        # for legacy we keep diffusion: use hi as extra
        target_len = len(prompt_words) + engine.rng.randint(lo, hi)
        result = engine.denoise(target_len=target_len, steps=int(args.steps), prompt=prompt_words, on_step=on_step)
        render_result(term, prompt_words, result)
        return 0
    # otherwise autocomplete
    return cmd_autocomplete(args, engine, term)


def cmd_chat(args: argparse.Namespace, engine: DiscreteDiffusionEngine, term: Term) -> int:
    print(term.paint("magenta", term.paint("bold", "\n› MLLM-5.2 Autocomplete — type a prefix, Tab to accept ghost.  :help  [quit] to exit.")))
    print(term.paint("dim", "  (causal n-gram, left context only — continues your document)\n"))
    while True:
        try:
            raw = input(term.paint("bold", "› ")).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not raw:
            continue
        low = raw.lower()
        if low in {"[quit]", "quit", "exit", ":quit", ":q"}:
            break
        if low in {":help", "help", "?"}:
            print(term.paint("dim", "  commands: [quit]/exit  :help  :clear  :steps N  :temp N  :seed N  :show on|off"))
            print(term.paint("dim", "  flags: --steps N --temperature T --threshold T --max-ngram N --seed N --corpus PATH"))
            print(term.paint("dim", "  playground: open index.html in browser for ghost-text editor"))
            continue
        if low in {":clear", "clear"}:
            os.system("clear" if os.name != "nt" else "cls")
            continue
        if low.startswith(":steps"):
            try:
                n = int(low.split()[1])
                if n < 1: raise ValueError
                args.steps = n
                print(term.paint("dim", f"  steps → {n}"))
            except Exception:
                print(term.paint("red", "  usage: :steps <positive int>"))
            continue
        if low.startswith(":temp"):
            try:
                t = float(low.split()[1])
                args.temperature = t
                print(term.paint("dim", f"  temperature → {t}"))
            except Exception:
                print(term.paint("red", "  usage: :temp <float 0.0-1.2>"))
            continue
        if low.startswith(":seed"):
            try:
                if low.split()[1].lower() == "none":
                    engine.rng = random.Random()
                    print(term.paint("dim", "  seed → random"))
                else:
                    s = int(low.split()[1])
                    engine.rng = random.Random(s)
                    print(term.paint("dim", f"  seed → {s}"))
            except Exception:
                print(term.paint("red", "  usage: :seed <int>  or  :seed none"))
            continue

        # treat input as prefix to autocomplete
        max_tokens = _get_max_tokens(args)
        prefix_words, result = do_autocomplete(engine, raw, max_tokens, float(getattr(args, "temperature", 0.35)), float(getattr(args, "threshold", 0.0)))
        render_autocomplete(term, prefix_words, result)
        print()
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    # ── Bare-prefix compatibility: support `python MLLM-5.2.py "hello world" --steps 4`
    # without requiring explicit subcommand, like ghost-text one-shot.
    # This keeps 5.1 architecture (BidirectionalTopology, DiscreteDiffusionEngine, etc.)
    # intact while making one-shot intuitive. If no known subcommand is present,
    # parse flags loosely and treat remaining as prompt (or stdin pipe).
    if argv is None:
        _argv_list = sys.argv[1:]
    else:
        _argv_list = list(argv)
    known_subs = {"autocomplete", "generate", "chat", "complete"}
    has_sub = any(tok in known_subs for tok in _argv_list)
    has_help = any(tok in ("-h", "--help", "--version") for tok in _argv_list)
    if not has_sub and not has_help:
        _tmp = argparse.ArgumentParser(add_help=False)
        _tmp.add_argument("--corpus", type=Path, default=None)
        _tmp.add_argument("--seed", type=int, default=None)
        _tmp.add_argument("--no-color", action="store_true")
        _tmp.add_argument("-v", "--verbose", action="store_true")
        _tmp.add_argument("--steps", type=int, default=16)
        _tmp.add_argument("--extra-tokens", type=int, nargs=2, default=None, metavar=("MIN", "MAX"))
        _tmp.add_argument("--temperature", type=float, default=0.35)
        _tmp.add_argument("--threshold", type=float, default=0.0)
        _tmp.add_argument("--max-ngram", type=int, default=3)
        _tmp.add_argument("--show-steps", action="store_true")
        try:
            _tmp_args, _remaining = _tmp.parse_known_args(_argv_list)
        except SystemExit:
            _tmp_args = None
            _remaining = None  # type: ignore
        if _tmp_args is not None and _remaining is not None:
            _unknown = [t for t in _remaining if t.startswith("-")]
            if not _unknown:
                _prompt_text = " ".join(_remaining).strip()
                _stdin_text = ""
                if not _prompt_text and not sys.stdin.isatty():
                    try:
                        _stdin_text = sys.stdin.read().strip()
                    except Exception:
                        _stdin_text = ""
                    if _stdin_text:
                        _prompt_text = _stdin_text
                # Only take bare path if there's a prompt or stdin or no args (REPL)
                # If _remaining is empty and no stdin, this is a bare REPL invocation -> handle here too
                # Distinguish from unknown-flag case which we already excluded
                # Proceed with bare handling (autocomplete or REPL)
                _seed = _tmp_args.seed
                _corpus_arg = _tmp_args.corpus
                _no_color = bool(_tmp_args.no_color)
                _verbose = bool(_tmp_args.verbose)
                logging.basicConfig(level=logging.DEBUG if _verbose else logging.WARNING,
                                    format="%(levelname)s %(name)s: %(message)s")
                _term = Term.detect(_no_color)
                if _term.enabled:
                    print(_term.paint("cyan", BANNER.strip("\n")))
                    print(_term.paint("dim", f"  v{__version__}  ·  autocomplete LM  ·  causal n={_tmp_args.max_ngram}  ·  ghost-text"))
                    print()
                else:
                    print(f"MLLM-5.2 v{__version__} — autocomplete LM")
                _extra = tuple(_tmp_args.extra_tokens) if _tmp_args.extra_tokens else None
                if _extra and len(_extra) == 2 and _extra[0] > _extra[1]:
                    print("mllm52: error: --extra-tokens MIN must be <= MAX", file=sys.stderr)
                    return 2
                _max_n = int(_tmp_args.max_ngram)
                _corpus_path = Path(_corpus_arg) if _corpus_arg else None
                try:
                    _topo = load_topology(_corpus_path, _term, max_n=_max_n)
                except (OSError, ValueError) as exc:
                    print(f"mllm52: error: cannot load corpus {_corpus_path or 'embedded'}: {exc}", file=sys.stderr)
                    return 2
                if _extra:
                    _tmp_args.extra_tokens = _extra
                else:
                    _tmp_args.extra_tokens = (int(_tmp_args.steps), int(_tmp_args.steps))
                _engine = DiscreteDiffusionEngine(_topo, rng=random.Random(_seed))
                _pseudo = argparse.Namespace(
                    prompt=_prompt_text if _prompt_text else None,
                    prefix=None,
                    steps=_tmp_args.steps,
                    extra_tokens=_tmp_args.extra_tokens,
                    temperature=_tmp_args.temperature,
                    threshold=_tmp_args.threshold,
                    max_ngram=_tmp_args.max_ngram,
                    show_steps=_tmp_args.show_steps,
                    seed=_seed,
                    corpus=_corpus_arg,
                    no_color=_no_color,
                    verbose=_verbose,
                    command="autocomplete" if _prompt_text else None,
                )
                if _prompt_text:
                    if _tmp_args.show_steps:
                        return cmd_generate(_pseudo, _engine, _term)
                    return cmd_autocomplete(_pseudo, _engine, _term)
                else:
                    # No prompt and no stdin -> REPL, but only if not already handled as help
                    # If argv_list was empty, this is plain REPL invocation
                    # Also if argv_list contained only flags, go to REPL with those flags
                    return cmd_chat(_pseudo, _engine, _term)
    parser = build_parser()
    args = parser.parse_args(argv)
    # allow global flags also after subcommand
    seed = getattr(args, "seed_sub", None) if getattr(args, "seed_sub", None) is not None else getattr(args, "seed", None)
    corpus_arg = getattr(args, "corpus_sub", None) if getattr(args, "corpus_sub", None) is not None else getattr(args, "corpus", None)
    no_color = bool(getattr(args, "no_color", False) or getattr(args, "no_color_sub", False))
    verbose = bool(getattr(args, "verbose", False) or getattr(args, "verbose_sub", False))

    logging.basicConfig(level=logging.DEBUG if verbose else logging.WARNING,
                        format="%(levelname)s %(name)s: %(message)s")
    term = Term.detect(no_color)

    if term.enabled:
        print(term.paint("cyan", BANNER.strip("\n")))
        print(term.paint("dim", f"  v{__version__}  ·  autocomplete LM  ·  causal n=3  ·  ghost-text"))
        print()
    else:
        print(f"MLLM-5.2 v{__version__} — autocomplete LM")



    # validate extra-tokens
    try:
        extra = tuple(args.extra_tokens) if getattr(args, "extra_tokens", None) else None
    except Exception:
        extra = None
    if extra and len(extra) == 2 and extra[0] > extra[1]:
        print("mllm52: error: --extra-tokens MIN must be <= MAX", file=sys.stderr)
        return 2

    max_n = int(getattr(args, "max_ngram", 3)) if hasattr(args, "max_ngram") else 3
    corpus_path = Path(corpus_arg) if corpus_arg else None
    try:
        topo = load_topology(corpus_path, term, max_n=max_n)
    except (OSError, ValueError) as exc:
        print(f"mllm52: error: cannot load corpus {corpus_path or 'embedded'}: {exc}", file=sys.stderr)
        return 2

    # patch args for downstream
    if extra:
        args.extra_tokens = extra
    else:
        # default extra_tokens = steps..steps for autocomplete
        args.extra_tokens = (int(getattr(args, "steps", 16)), int(getattr(args, "steps", 16)))
    if not hasattr(args, "steps") or args.steps is None:
        args.steps = 16
    if not hasattr(args, "temperature"):
        args.temperature = 0.35
    if not hasattr(args, "threshold"):
        args.threshold = 0.0

    engine = DiscreteDiffusionEngine(topo, rng=random.Random(seed))
    cmd = getattr(args, "command", None)
    if cmd in ("autocomplete", "complete"):
        # Let cmd_autocomplete handle prompt or stdin or fallback to REPL
        return cmd_autocomplete(args, engine, term)
    if cmd is None and getattr(args, "prefix", None) is None:
        # No subcommand and no prefix attribute -> default REPL (or bare already handled)
        # Check stdin for one-shot without subcommand (should have been handled earlier, but keep)
        if not sys.stdin.isatty():
            try:
                _stdin_prompt = sys.stdin.read().strip()
            except Exception:
                _stdin_prompt = ""
            if _stdin_prompt:
                _pseudo = argparse.Namespace(prompt=_stdin_prompt, prefix=None, steps=getattr(args, "steps", 16),
                                             extra_tokens=getattr(args, "extra_tokens", None),
                                             temperature=getattr(args, "temperature", 0.35),
                                             threshold=getattr(args, "threshold", 0.0),
                                             max_ngram=getattr(args, "max_ngram", 3),
                                             show_steps=getattr(args, "show_steps", False))
                return cmd_autocomplete(_pseudo, engine, term)
        return cmd_chat(args, engine, term)
    if cmd == "generate":
        return cmd_generate(args, engine, term)
    if cmd == "chat":
        return cmd_chat(args, engine, term)
    return cmd_chat(args, engine, term)


if __name__ == "__main__":
    raise SystemExit(main())
