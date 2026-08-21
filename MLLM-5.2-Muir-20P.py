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
BUILT_IN_CORPUS = """
1 + 1 = 2. 2 + 2 = 4. 3 + 3 = 6. 4 + 4 = 8. 5 + 5 = 10. 6 + 6 = 12. 7 + 7 = 14. 8 + 8 = 16. 9 + 9 = 18. 10 + 10 = 20. 11 + 11 = 22. 12 + 12 = 24. 13 + 13 = 26. 14 + 14 = 28. 15 + 15 = 30. 16 + 16 = 32. 17 + 17 = 34. 18 + 18 = 36. 19 + 19 = 38. 20 + 20 = 40. 25 + 25 = 50. 30 + 30 = 60. 35 + 35 = 70. 40 + 40 = 80. 45 + 45 = 90. 50 + 50 = 100. 1 * 1 = 1. 2 * 2 = 4. 3 * 3 = 9. 4 * 4 = 16. 5 * 5 = 25. 6 * 6 = 36. 7 * 7 = 49. 8 * 8 = 64. 9 * 9 = 81. 10 * 10 = 100. 2 * 1 = 2. 2 * 2 = 4. 2 * 3 = 6. 2 * 4 = 8. 2 * 5 = 10. 2 * 6 = 12. 2 * 7 = 14. 2 * 8 = 16. 2 * 9 = 18. 2 * 10 = 20. 3 * 1 = 3. 3 * 2 = 6. 3 * 3 = 9. 3 * 4 = 12. 3 * 5 = 15. 3 * 6 = 18. 3 * 7 = 21. 3 * 8 = 24. 3 * 9 = 27. 3 * 10 = 30. 4 * 1 = 4. 4 * 2 = 8. 4 * 3 = 12. 4 * 4 = 16. 4 * 5 = 20. 4 * 6 = 24. 4 * 7 = 28. 4 * 8 = 32. 4 * 9 = 36. 4 * 10 = 40. Earth is the third planet from the Sun and the only astronomical object known to harbor life. This is made possible by Earth being an ocean world, the only one in the Solar System sustaining liquid surface water. Almost all of Earth's water is contained in its global ocean, covering 70.8% of Earth's crust. The remaining 29.2% of Earth's crust is land, most of which is located in the form of continental landmasses within Earth's land hemisphere. Most of Earth's land is at least somewhat humid and covered by vegetation, while large ice sheets at Earth's polar deserts retain more water than Earth's groundwater, lakes, rivers, and atmospheric water combined. Earth's crust consists of slowly moving tectonic plates, which interact to produce mountain ranges, volcanoes, and earthquakes. Earth has a liquid outer core that generates a magnetosphere capable of deflecting most of the destructive solar winds and cosmic radiation. Science is a systematic discipline that builds and organises knowledge in the form of testable hypotheses and predictions about the universe. Modern science is typically divided into two – or three – major branches: the natural sciences, which study the physical world, and the social sciences, which study individuals and societies. While referred to as the formal sciences, the study of logic, mathematics, and theoretical computer science are typically regarded as separate because they rely on deductive reasoning instead of the scientific method as their main methodology. Meanwhile, applied sciences are disciplines that use scientific knowledge for practical purposes, such as engineering and medicine. Artificial intelligence (AI) is the capability of computational systems to perform tasks typically associated with human intelligence, such as learning, reasoning, problem-solving, perception, and decision-making. It is a field of research in computer science that develops and studies methods and software that enable machines to perceive their environment and use learning and intelligence to take actions that maximize their chances of achieving defined goals. A computer is a machine that can be programmed to automatically carry out sequences of arithmetic or logical operations (computation). Modern digital electronic computers can perform generic sets of operations known as programs, which enable computers to perform a wide range of tasks. The term computer system may refer to a nominally complete computer that includes the hardware, operating system, software, and peripheral equipment needed and used for full operation, or to a group of computers that are linked and function together, such as a computer network or computer cluster. A school is an educational institution designed to provide learning environments for the teaching of students, usually under the direction of teachers. Most countries have systems of formal education, which is sometimes compulsory. In these systems, students progress through a series of schools that can be built and operated by both government and private organizations. The names for these schools vary by country but generally include primary school for young children and secondary school for teenagers who have completed primary education. An institution where higher education is taught is commonly called a university college or university. Cozmo is a miniature robot created by the defunct company Anki. Cozmo's base model, is a small, white and gray robot with red highlights. It makes use of distinct expressions, dubbed the "emotion engine", in order to mimic human emotion. Later editions came in red and white, gray and black and another in blue. 

Physics is the scientific study of matter, its fundamental constituents, its motion and behavior through space and time, and the related entities of energy and force. It is one of the most fundamental scientific disciplines. A scientist who specializes in the field of physics is called a physicist. Chemistry is the scientific study of the properties and behavior of matter. It is a physical science within the natural sciences that studies the chemical elements that make up matter and compounds made of atoms, molecules and ions: their composition, structure, properties, behavior and the changes they undergo during reactions with other substances. Chemistry also addresses the nature of chemical bonds in chemical compounds. Biology is the scientific study of life and living organisms. It is a broad natural science that encompasses a wide range of fields and unifying principles that explain the structure, function, growth, origin, evolution, and distribution of life. Central to biology are five fundamental themes: the cell as the basic unit of life, genes and heredity as the basis of inheritance, evolution as the driver of biological diversity, energy transformation for sustaining life processes, and the maintenance of internal stability (homeostasis). Astronomy is a natural science that studies celestial objects and the phenomena that occur in the cosmos. It uses mathematics, physics, and chemistry to explain their origin and their overall evolution. Objects of interest include planets, moons, stars, nebulae, galaxies, meteoroids, asteroids, and comets. Relevant phenomena include supernova explosions, gamma ray bursts, quasars, blazars, pulsars, and cosmic microwave background radiation. More generally, astronomy studies everything that originates beyond Earth's atmosphere. Cosmology is the branch of astronomy that studies the universe as a whole. Geology is a branch of natural science concerned with the Earth and other astronomical bodies, the rocks of which they are composed, and the processes by which they change over time. The name comes from Ancient Greek γῆ (gê) 'earth' and λoγία (-logía) 'study of, discourse'. Modern geology significantly overlaps all other Earth sciences, including hydrology. It is integrated with Earth system science and planetary science. Ecology is the natural science of the relationships among living organisms and their environment. Ecology considers organisms at the individual, population, community, ecosystem, and biosphere levels. Ecology overlaps with the closely related sciences of biogeography, evolutionary biology, genetics, ethology, and natural history. Mathematics is a field of study that discovers and organizes methods, theories, and theorems that are developed and proved for the needs of empirical sciences and mathematics itself. There are many areas of mathematics, which include number theory, algebra, geometry, analysis, and set theory. Statistics is the discipline that concerns the collection, organization, analysis, interpretation, and presentation of data. In applying statistics to a scientific, industrial, or social problem, it is conventional to begin with a statistical population or a statistical model to be studied. Populations can be diverse groups of people or objects such as "all people living in a country" or "every atom composing a crystal". Statistics deals with every aspect of data, including the planning of data collection in terms of the design of surveys and experiments. Logic is the study of correct reasoning. It includes both formal and informal logic. Formal logic is the study of deductively valid inferences or logical truths. It examines how conclusions follow from premises based on the structure of arguments alone, independent of their topic and content. Informal logic is associated with informal fallacies, critical thinking, and argumentation theory. Informal logic examines arguments expressed in natural language whereas formal logic uses formal language. When used as a countable noun, the term "a logic" refers to a specific logical formal system that articulates a proof system. Logic plays a central role in many fields, such as philosophy, mathematics, computer science, and linguistics. Philosophy is a systematic study of general and fundamental questions concerning topics like existence, knowledge, mind, reason, language, and value. It is a rational and critical inquiry that reflects on its methods and assumptions. Psychology is the scientific study of the mind and behavior. Its subject matter includes the behavior of humans and nonhumans, both conscious and unconscious phenomena, and mental processes such as thoughts, feelings, and motives. Psychology is an academic discipline of immense scope, crossing the boundaries between the natural and social sciences. Biological psychologists seek an understanding of the emergent properties of brains, linking the discipline to neuroscience. As social scientists, psychologists aim to understand the behavior of individuals and groups. Sociology is the scientific study of human society that focuses on society, human social behavior, patterns of social relationships, social interaction, and aspects of culture associated with everyday life. The term sociology was coined in the late 18th century to describe the scientific study of society. Regarded as a part of both the social sciences and humanities, sociology uses various methods of empirical investigation and critical analysis to develop a body of knowledge about social order and social change. Sociological subject matter ranges from micro-level analyses of individual interaction and agency to macro-level analyses of social systems and social structure. Applied sociological research may be applied directly to social policy and welfare, whereas theoretical approaches may focus on the understanding of social processes and phenomenological method. Economics is a social science that studies the production, distribution, and consumption of goods and services. Political science is the social scientific study of politics. It deals with systems of governance and power, and the analysis of political activities, political thought, political behavior, and associated constitutions and laws. Specialists in the field are political scientists. Unlike political philosophy, which is primarily normative and concerns the theoretical and conceptual foundations of politics, political science emphasizes descriptive and explanatory of what is and favors empirical evidence over ethical judgements. History is the systematic study of the past, focusing primarily on the human past. As an academic discipline, it analyses and interprets evidence to construct narratives about what happened and explain why it happened. Some theorists categorize history as a social science, while others see it as part of the humanities or consider it a hybrid discipline. Similar debates surround the purpose of history—for example, whether its main aim is theoretical, to uncover the truth, or practical, to learn lessons from the past. In a more general sense, the term history refers not to an academic field but to the past itself, times in the past, or to individual texts about the past. Archaeology or archeology is the study of human activity through the recovery and analysis of material culture. The archaeological record consists of artifacts, architecture, biofacts or ecofacts, sites, and cultural landscapes. Archaeology can be considered both a social science and a branch of the humanities. It is usually considered an independent academic discipline, but may also be classified as part of anthropology, history or geography. The discipline involves surveying, excavation, and eventually analysis of data collected, to learn more about the past. In broad scope, archaeology relies on cross-disciplinary research. Anthropology is the scientific study of humanity that crosses biology and sociology, concerned with human behavior, human biology, cultures, societies, and linguistics, in both the present and past, including archaic humans. Social anthropology studies patterns of behaviour, while cultural anthropology studies cultural meaning, including norms and values. The term sociocultural anthropology is commonly used today. Linguistic anthropology studies how language influences social life. Biological anthropology studies the biology and evolution of humans and their close primate relatives. Linguistics is the scientific study of language. The areas of linguistic analysis are syntax, semantics (meaning), morphology, phonetics, phonology, and pragmatics. Subdisciplines such as biolinguistics and psycholinguistics bridge many of these divisions. Literature is any collection of written work. The term is also used more narrowly for writings considered an art form, especially novels, plays, and poems. It includes both print and digital writing. In recent centuries, the definition has expanded to include oral literature, much of which has been transcribed. Literature is a method of recording, preserving, and transmitting knowledge and entertainment. It can also have a social, psychological, spiritual, or political role. Art is a diverse range of cultural activity centered around works utilizing creative or imaginative talents, which are expected to evoke a worthwhile experience, generally through an expression of emotional power, conceptual ideas, technical proficiency, or beauty. Music theory is the study of theoretical frameworks for understanding the practices and possibilities of music. The Oxford Companion to Music describes three interrelated uses of the term "music theory": The first refers to the "rudiments" needed to understand music notation such as key signatures, time signatures, and rhythmic notation; the second is a study of scholars' views on music from antiquity to the present; the third is a sub-topic of musicology that "seeks to define processes and general principles in music". The musicological approach to theory differs from musical analysis "in that it takes as its starting-point not the individual work or performance but the fundamental materials from which it is built.". Engineering is the practice of using natural science, mathematics, and the engineering design process to solve problems within technology, increase efficiency and productivity, and improve systems. The traditional disciplines of engineering are civil, mechanical, electrical, and chemical. The academic discipline of engineering encompasses a broad range of more specialized subfields, and each can have a more specific emphasis for applications of mathematics and science. In turn, modern engineering practice spans multiple fields of engineering, which include designing and improving infrastructure, machinery, vehicles, electronics, materials, and energy systems. For related terms, see glossary of engineering. Electrical engineering is an engineering discipline concerned with the study, design, and application of equipment, devices, and systems that use electricity, electronics, and electromagnetism. It emerged as an identifiable occupation in the latter half of the 19th century after the commercialization of the electric telegraph, the telephone, and electrical power generation, distribution, and use. The American Society of Mechanical Engineers (ASME) is an American professional association that, in its own words, "promotes the art, science, and practice of multidisciplinary engineering and allied sciences around the globe" via "continuing education, training and professional development, codes and standards, research, conferences and publications, government relations, and other forms of outreach." ASME is thus an engineering society, a standards organization, a research and development organization, an advocacy organization, a provider of training and education, and a nonprofit organization. Founded as an engineering society focused on mechanical engineering in North America, ASME is today multidisciplinary and global. Civil Engineering is a professional engineering discipline that deals with the design, construction, and maintenance of the physical and naturally built environment, including public works such as roads, bridges, canals, dams, airports, sewage systems, pipelines, structural components of buildings, and railways. Computer science is the study of computation, information, and automation. Included broadly in the sciences, computer science spans theoretical disciplines to applied disciplines. An expert in the field is known as a computer scientist. Computer security is a subdiscipline within the field of information security. It focuses on protecting computer software, systems, and networks from threats that can lead to unauthorized information disclosure, theft or damage to hardware, software, or data, as well as to the disruption or misdirection of the services they provide. Data science is an interdisciplinary academic field that uses statistics, scientific computing, scientific methods, processing, scientific visualization, algorithms, and systems to extract or extrapolate knowledge from potentially noisy, structured, or unstructured data. Machine learning (ML) is a field of study in artificial intelligence concerned with the development and study of statistical algorithms that can learn from data and generalize to unseen data, and thus perform tasks without explicit instructions. Within a subdiscipline in machine learning, advances in the field of deep learning have allowed neural networks, a class of statistical algorithms, to surpass many previous machine learning approaches in performance. 

An operating system (OS) is system software that manages computer hardware and software resources, and provides common services for computer programs. Cloud computing is defined by the ISO as "a paradigm for enabling network access to a scalable and elastic pool of shareable physical or virtual resources with self-service provisioning and administration on demand". It is commonly referred to as "the cloud". In computing, a database is an organized collection of data or a type of data store based on the use of a database management system (DBMS), the software that interacts with end users, applications, and the database itself to capture and analyze the data. The DBMS additionally encompasses the core facilities provided to administer the database. The sum total of the database, the DBMS and the associated applications can be referred to as a database system. Often the term "database" is also used loosely to refer to any of the DBMS, the database system or an application associated with the database. Genetics is the study of genes, genetic variation, and heredity in organisms. It is an important branch in biology because heredity is vital to organisms' evolution. Gregor Mendel, a Moravian Augustinian friar working in the 19th century in Brno, was the first to study genetics scientifically. Mendel studied "trait inheritance", patterns in the way traits are handed down from parents to offspring over time. He observed that organisms inherit traits by way of discrete "units of inheritance". This term, still used today, is a somewhat ambiguous definition of what is referred to as a gene. Neuroscience is the scientific study of the nervous system, its functions, and its disorders. It is a multidisciplinary science that combines physiology, anatomy, molecular biology, developmental biology, cytology, psychology, physics, computer science, chemistry, medicine, statistics, and mathematical modeling to understand the fundamental and emergent properties of neurons, glia, and neural circuits. The understanding of the biological basis of learning, memory, behavior, perception, and consciousness has been described by Eric Kandel as the "epic challenge" of the biological sciences. Medicine is the science and practice of caring for patients, managing the diagnosis, prognosis, prevention, treatment and palliation of their injury or disease, while promoting their health. Medicine encompasses a variety of health care practices which evolved to maintain and restore health through the prevention and treatment of illness. Contemporary medicine applies biomedical sciences, biomedical research, genetics, and medical technology to diagnose, treat, and prevent injury and disease, typically through various pharmaceuticals or surgery, but also through therapies such as psychotherapy, external splints and traction, medical devices, biologics, and ionizing radiation, amongst others. Public Health may refer to:Public health, promoting health through organized efforts and informed choices of society and individuals Public Health (journal), published by Elsevier for the Royal Society for Public Health Public Health a 2021 proposed comedy television series by Rob Tepper Public Health, a May 22, 2014 episode of Debatten, a Norwegian television series Public Health, a July 6, 2000 episode of Today's Environment, television series by Five Star Productions. Law is a set of rules that are created and are enforceable by governmental or societal institutions to regulate behavior, with its precise definition a matter of longstanding debate. It has been variously described as a science and as the art of justice. State-enforced laws can be made by a legislature, resulting in statutes; by the executive through decrees and regulations; or by judges' decisions, which form precedent in common law jurisdictions. An autocrat may exercise those functions within their realm. The creation of laws themselves may be influenced by a constitution, written or tacit, and the rights encoded therein. The law shapes politics, economics, history and society in various ways and also serves as a mediator of relations between people. Ethics is the philosophical study of moral phenomena. Also called moral philosophy, it investigates normative questions about what people ought to do or which behavior is morally right. Its main branches include normative ethics, applied ethics, and metaethics. Business is the practice of making one's living or making money by producing or buying and selling products. It is also "any activity or enterprise entered into for profit.". Finance refers to monetary resources and to the study and discipline of money, currency, assets and liabilities. As a subject of study, it is a field of business administration which involves the planning, organizing, leading, and controlling of an organization's resources to achieve its goals. Based on the scope of financial activities in financial systems, the discipline can be divided into personal, corporate, and public finance. Marketing is the act of acquiring, satisfying and retaining customers. It is one of the primary components of business management and commerce. Entrepreneurship is the creation or extraction of economic value by identifying and commercializing opportunities to deliver products or services, a process that typically requires considerable initiative and bears risk. This process may also encompass the pursuit of values that extend beyond mere economic considerations. Geopolitics is the study of the effects of Earth's geography on politics and international relations. Geopolitics usually refers to countries and relations between them. According to multiple researchers, the term is currently being used to describe a broad spectrum of concepts, in a general sense used as "a synonym for international political relations", but more specifically "to imply the global structure of such relations"; this usage builds on an "early-twentieth-century term for a pseudoscience of political geography" and other pseudoscientific theories of historical and geographic determinism. Climatology or climate science is the scientific study of Earth's climate, typically defined as weather conditions averaged over a period of at least 30 years. Climate concerns the atmospheric condition during an extended to indefinite period of time; weather is the condition of the atmosphere during a relative brief period of time. The main topics of research are the study of climate variability, mechanisms of climate changes and modern climate change. This topic of study is regarded as part of the atmospheric sciences and a subdivision of physical geography, which is one of the Earth sciences. Climatology includes some aspects of oceanography and biogeochemistry. Failed to fetch Energy Systems. Environmental science is an academic field that integrates the physical, biological, and mathematical sciences to study the environment and solve environmental problems. It uses an integrated, quantitative, and interdisciplinary approach to analyze environmental systems and emerged from the fields of natural history and medicine during the Enlightenment. It is considered interdisciplinary because it is an integration of various fields such as: biology, chemistry, physics, geology, engineering, sociology, and ecology. Astronautics is the practice of sending spacecraft beyond Earth's atmosphere into outer space. Spaceflight is one of its main applications and space science is its overarching field. Robotics is the interdisciplinary study and practice of the design, construction, operation, and use of robots. A roboticist is someone who specializes in robotics. Automation describes a wide range of technologies that reduce human intervention in processes, mainly by predetermining decision criteria, subprocess relationships, and related actions, as well as embodying those predeterminations in machines. Automation has been achieved by various means including mechanical, hydraulic, pneumatic, electrical, electronic devices, and computers, usually in combination. Complicated systems, such as modern factories, airplanes, and ships typically use combinations of all of these techniques. The benefits of automation includes labor savings, reducing waste, savings in electricity costs, savings in material costs, and improvements to quality, accuracy, and precision. Biotechnology is a multidisciplinary field that involves the integration of natural sciences and engineering sciences in order to achieve the application of organisms and parts thereof for products and services. Specialists in the field are known as biotechnologists. Nanotechnology is the manipulation of matter with at least one dimension sized from 1 to 100 nanometers (nm). At this scale, commonly known as the nanoscale, surface area and quantum mechanical effects become important in describing properties of matter. This definition of nanotechnology includes all types of research and technologies that deal with these special properties. It is common to see the plural form "nanotechnologies" as well as "nanoscale technologies" to refer to research and applications whose common trait is scale. An earlier understanding of nanotechnology referred to the particular technological goal of precisely manipulating atoms and molecules for fabricating macroscale products, now referred to as molecular nanotechnology. Materials science is an interdisciplinary field of researching and discovering materials. Materials engineering is an engineering field of finding uses for materials in other fields and industries. Cognitive science is the interdisciplinary, scientific study of the mind and its processes. It examines the nature, the tasks, and the functions of cognition. Mental faculties of concern to cognitive scientists include perception, memory, attention, reasoning, language, and emotion. To understand these faculties, cognitive scientists borrow from fields such as psychology, philosophy, artificial intelligence, neuroscience, linguistics, and anthropology. The typical analysis of cognitive science spans many levels of organization, from learning and decision-making to logic and planning; from neural circuitry to modular brain organization. One of the fundamental concepts of cognitive science is that "thinking can best be understood in terms of representational structures in the mind and computational procedures that operate on those structures.". Game theory is the study of mathematical models of strategic interactions. It has applications in many fields of social science, and is used extensively in economics, logic, systems science and computer science. Initially, game theory addressed two-person zero-sum games, in which a participant's gains or losses are exactly balanced by the losses and gains of the other participant. In the 1950s, it was extended to the study of non zero-sum games, and was eventually applied to a wide range of behavioral relations. It is now an umbrella term for the science of rational decision making in humans, animals, and computers. Information theory is the mathematical study of the quantification, storage, and communication of a particular type of mathematically defined information. The field was established and formalized by Claude Shannon in the 1940s, though early contributions were made in the 1920s through the works of Harry Nyquist and Ralph Hartley. It is at the intersection of electronic engineering, mathematics, statistics, computer science, neurobiology, physics, and electrical engineering. Employment is a relationship between two parties regulating the provision of paid labour services. Usually based on a contract, one party, the employer, which might be a corporation, a not-for-profit organization, a co-operative, or any other entity, pays the other, the employee, in return for carrying out assigned work. Employees work in return for wages, which can be paid on the basis of an hourly rate, by piecework or an annual salary, depending on the type of work an employee does, the prevailing conditions of the sector and the bargaining power between the parties. Employees in some sectors may receive gratuities, bonus payments or stock options. In some types of employment, employees may receive benefits in addition to payment. Benefits may include health insurance, housing, and disability insurance. Education is the transmission of knowledge and skills and the development of character traits. Formal education happens in a complex institutional framework, like public schools. Non-formal education is also structured but takes place outside the formal schooling system, while informal education is unstructured learning through daily experiences. Formal and non-formal education are divided into levels that include early childhood education, primary education, secondary education, and tertiary education. Other classifications focus on the teaching method, like teacher-centered and student-centered education, and on the subject, like science education, language education, and physical education. The term "education" can also refer to the mental states and qualities of educated people and the academic field studying educational phenomena. Sleep is a state of reduced mental and physical activity in which consciousness is altered and certain sensory activity is inhibited. During sleep, there is a marked decrease in muscle activity and interactions with the surrounding environment. While sleep differs from wakefulness in terms of the ability to react to stimuli, it still involves active brain patterns, making it more reactive than a coma or disorders of consciousness. A hobby is considered to be a regular activity that is done for enjoyment, typically during one's leisure time. Hobbies include collecting themed items and objects, engaging in creative and artistic pursuits, playing sports, or pursuing other amusements or avocations. Participation in hobbies encourages acquiring substantial skills and knowledge in that area. A list of hobbies changes with renewed interests and developing fashions, making it diverse and lengthy. Hobbies tend to follow trends in society. For example, stamp collecting was popular during the nineteenth and twentieth centuries as postal systems were the main means of communication; as of 2024, video games became more popular following technological advances. The advancing production, technology, and labour movements of the nineteenth century provided workers with more leisure time to engage in hobbies. Because of this, the efforts of people investing in hobbies has increased with time. Shopping is an activity in which a customer browses the available goods or services presented by one or more retailers with the potential intent to purchase a suitable selection of them. A typology of shopper types has been developed by scholars which identifies one group of shoppers as recreational shoppers, that is, those who enjoy shopping and view it as a leisure activity. Health has a variety of definitions, which have been used for different purposes over time. In general, it refers to physical and emotional well-being, especially that associated with normal functioning of the human body, absent of disease, pain, or injury. Family is a group of people related either by consanguinity or affinity. It forms the basis for social order. Ideally, families offer predictability, structure, and safety as members mature and learn to participate in the community. Historically, most human societies use family as the primary purpose of attachment, nurturance, and socialization. Leisure has often been defined as a quality of experience or as free time. Free time is time spent away from business, work, job hunting, domestic chores, and education, as well as necessary activities such as eating and sleeping. Leisure as an experience usually emphasizes dimensions of perceived freedom and choice. It is done for "its own sake", for the quality of experience and involvement. Other classic definitions include Thorstein Veblen's (1899) of "nonproductive consumption of time." Free time is not easy to define due to the multiplicity of approaches used to determine its essence. Different disciplines have definitions reflecting their common issues: for example, sociology on social forces and contexts and psychology as mental and emotional states and conditions. From a research perspective, these approaches have an advantage of being quantifiable and comparable over time and place. A grocery store (AE), grocery shop or grocer's shop (BE) or simply grocery is a retail store that primarily retails a general range of food products, which may be fresh or packaged. In everyday US usage, however, "grocery store" is a synonym for supermarket, and is not used to refer to other types of stores that sell groceries. In the UK, shops that sell food are distinguished as grocers or grocery shops. Physical fitness is a state of health and well-being and, more specifically, the ability to perform aspects of sports, occupations, and daily activities. Physical fitness is generally achieved through proper nutrition, moderate-vigorous physical exercise, and sufficient rest along with a formal recovery plan. Cleaning is the process of removing unwanted substances, such as dirt, dust, and other impurities, from an object or environment. Cleaning is often performed for aesthetic, hygienic, functional, safety, or environmental protection purposes. Cleaning occurs in many different contexts, and uses many different methods. Several occupations are devoted to cleaning. Laundry is the washing of clothing and other textiles, and, more broadly, their drying and ironing as well. Laundry has been part of history since humans began to wear clothes, so the methods by which different cultures have dealt with this universal human need are of interest to several branches of scholarship. Personal finance is the financial management that an individual or a family unit performs to budget, save, and spend monetary resources in a controlled manner, taking into account various financial risks and future life events. Telecommunication, often used in its plural form or abbreviated as telecom, is the transmission of information over a distance using electrical or electronic means, typically through cables, radio waves, or other communication technologies. These means of transmission may be divided into communication channels for multiplexing, allowing for a single medium to transmit several concurrent communication sessions. Long-distance technologies invented during the 19th, 20th and 21st centuries generally use electric power, and include the electrical telegraph, telephone, television, and radio. Social media are new media technologies that facilitate the creation, sharing and aggregation of content amongst virtual communities and networks. Common features include:Online platforms enable users to create and share content and participate in social networking. User-generated content—such as text posts or comments, digital photos or videos, and data generated through online interactions. Service-specific profiles that are designed and maintained by the social media organization. Social media helps the development of online social networks by connecting a user's profile with those of other individuals or groups. Television (TV) is a telecommunication medium for transmitting moving images and sound. Additionally, the term can refer to a physical television set rather than the medium of transmission. Television is a mass medium for advertising, entertainment, news, and sports. The medium is capable of more than "radio broadcasting", which refers to an audio signal sent to radio receivers. A birthday is the anniversary of the birth of a person or the figurative birth of an institution. Birthdays of people are celebrated in numerous cultures, often with birthday gifts, birthday cards, a birthday party, or a rite of passage. A wedding is a ceremony in which two people are united in marriage. Wedding traditions and customs vary greatly between cultures, ethnicities, races, religions, denominations, countries, social classes, and sexual orientations. Most wedding ceremonies involve an exchange of marriage vows by a couple; a presentation of a gift ; and a public proclamation of marriage by an authority figure or celebrant. Special wedding garments are often worn, and the ceremony is sometimes followed by a wedding reception. Music, poetry, prayers, or readings from religious texts or literature are also commonly incorporated into the ceremony, as well as superstitious customs. A funeral is a ceremony connected with the final disposition of a corpse, such as a burial, entombment or cremation with the attendant observances. Funerary customs comprise the complex of beliefs and practices used by a culture to remember and respect the dead, from interment, to various monuments, prayers, and rituals undertaken in their honour. Customs vary between cultures and religious groups. Funerals have both normative and legal components. Common secular motivations for funerals include mourning the deceased, celebrating their life, and offering support and sympathy to the bereaved; additionally, funerals may have religious aspects that are intended to help the soul of the deceased reach the afterlife, resurrection or reincarnation. Religion is a range of social-cultural systems, including designated behaviors and practices, ethics, morals, beliefs, worldviews, texts, sanctified places, prophecies, or organizations, that generally relate humanity to supernatural, transcendental, and spiritual elements—although there is no scholarly consensus over what precisely constitutes a religion. It is an essentially contested concept. Different religions may or may not contain various elements ranging from the divine, sacredness, faith, and a supernatural being or beings. Politics is the set of activities that are associated with making decisions in groups, or other forms of power relations among individuals, such as the distribution of status or resources. The branch of social science that studies politics and government is referred to as political science. History is the systematic study of the past, focusing primarily on the human past. As an academic discipline, it analyses and interprets evidence to construct narratives about what happened and explain why it happened. Some theorists categorize history as a social science, while others see it as part of the humanities or consider it a hybrid discipline. Similar debates surround the purpose of history—for example, whether its main aim is theoretical, to uncover the truth, or practical, to learn lessons from the past. In a more general sense, the term history refers not to an academic field but to the past itself, times in the past, or to individual texts about the past. Geography is the study of the lands, features, inhabitants, and phenomena of Earth. Geography is an all-encompassing discipline that seeks an understanding of Earth and its human and natural complexities—not merely where objects are, but also how they have changed and come to be. While geography is specific to Earth, many concepts can be applied more broadly to other celestial bodies in the field of planetary science. Geography has been called "a bridge between natural science and social science disciplines.". Science is a systematic discipline that builds and organises knowledge in the form of testable hypotheses and predictions about the universe. Modern science is typically divided into two – or three – major branches: the natural sciences, which study the physical world, and the social sciences, which study individuals and societies. While referred to as the formal sciences, the study of logic, mathematics, and theoretical computer science are typically regarded as separate because they rely on deductive reasoning instead of the scientific method as their main methodology. Meanwhile, applied sciences are disciplines that use scientific knowledge for practical purposes, such as engineering and medicine. Technology is the application of conceptual knowledge to achieve practical goals, especially in a reproducible way. The word technology can also mean the products resulting from such efforts, including both tangible tools such as utensils or machines, and intangible ones such as software. Technology plays a critical role in science, engineering, and everyday life. Art is a diverse range of cultural activity centered around works utilizing creative or imaginative talents, which are expected to evoke a worthwhile experience, generally through an expression of emotional power, conceptual ideas, technical proficiency, or beauty. Literature is any collection of written work. The term is also used more narrowly for writings considered an art form, especially novels, plays, and poems. It includes both print and digital writing. In recent centuries, the definition has expanded to include oral literature, much of which has been transcribed. Literature is a method of recording, preserving, and transmitting knowledge and entertainment. It can also have a social, psychological, spiritual, or political role. Philosophy is a systematic study of general and fundamental questions concerning topics like existence, knowledge, mind, reason, language, and value. It is a rational and critical inquiry that reflects on its methods and assumptions. Psychology is the scientific study of the mind and behavior. Its subject matter includes the behavior of humans and nonhumans, both conscious and unconscious phenomena, and mental processes such as thoughts, feelings, and motives. Psychology is an academic discipline of immense scope, crossing the boundaries between the natural and social sciences. Biological psychologists seek an understanding of the emergent properties of brains, linking the discipline to neuroscience. As social scientists, psychologists aim to understand the behavior of individuals and groups. Sociology is the scientific study of human society that focuses on society, human social behavior, patterns of social relationships, social interaction, and aspects of culture associated with everyday life. The term sociology was coined in the late 18th century to describe the scientific study of society. Regarded as a part of both the social sciences and humanities, sociology uses various methods of empirical investigation and critical analysis to develop a body of knowledge about social order and social change. Sociological subject matter ranges from micro-level analyses of individual interaction and agency to macro-level analyses of social systems and social structure. Applied sociological research may be applied directly to social policy and welfare, whereas theoretical approaches may focus on the understanding of social processes and phenomenological method. Economics is a social science that studies the production, distribution, and consumption of goods and services. Law is a set of rules that are created and are enforceable by governmental or societal institutions to regulate behavior, with its precise definition a matter of longstanding debate. It has been variously described as a science and as the art of justice. State-enforced laws can be made by a legislature, resulting in statutes; by the executive through decrees and regulations; or by judges' decisions, which form precedent in common law jurisdictions. An autocrat may exercise those functions within their realm. The creation of laws themselves may be influenced by a constitution, written or tacit, and the rights encoded therein. The law shapes politics, economics, history and society in various ways and also serves as a mediator of relations between people. An apple is the round, edible fruit of an apple tree. Fruit trees of the orchard or domestic apple, the most widely grown in the genus, are cultivated worldwide. The tree originated in Central Asia, where its wild ancestor, Malus sieversii, is still found. Apples have been grown for thousands of years in Eurasia before they were introduced to North America by European colonists. Apples have cultural significance in many mythologies and religions. A banana is an elongated, edible fruit—botanically a berry—produced by several kinds of large treelike herbaceous flowering plants in the genus Musa. In some countries, cooking bananas are called plantains, distinguishing them from dessert bananas. The fruit is variable in size, color and firmness, but is usually elongated and curved, with soft flesh rich in starch covered with a peel, which may have a variety of colors when ripe. It grows upward in clusters near the top of the plant. Almost all modern edible seedless (parthenocarp) cultivated bananas come from two wild species – Musa acuminata and Musa balbisiana, or their hybrids. Orange most often refers to:Orange (fruit), the fruit of the tree species Citrus × sinensis Orange blossom, its fragrant flower Orange juice Orange (colour), the color of an orange fruit, occurs between red and yellow in the visible light The tomato is a plant whose fruit is an edible berry that is eaten as a vegetable. The tomato is a member of the nightshade family that includes tobacco, potato, and chili peppers. It originated from western South America, and may have been domesticated there, in Mexico, or in Central America. The Spanish introduced tomatoes to Eurasia in the Columbian exchange in the 16th century. The cucumber is a widely-cultivated creeping vine plant in the family Cucurbitaceae that bears cylindrical to spherical fruits, which are used as culinary vegetables. Considered an annual plant, there are three main types of cucumber—slicing, pickling, and seedless—within which several cultivars have been created. The cucumber originates in Asia extending from India, Nepal, Bangladesh, China, and Northern Thailand, but now grows on most continents, and many different types of cucumber are grown commercially and traded on the global market. In North America, the term wild cucumber refers to plants in the genera Echinocystis and Marah, though the two are not closely related. The carrot is a root vegetable, typically orange in colour, though heirloom variants including purple, black, red, white, and yellow cultivars exist, all of which are domesticated forms of the wild carrot, Daucus carota, native to Europe and Southwestern Asia. The plant probably originated in Iran and was originally cultivated for its leaves and seeds. Broccoli is an edible green plant in the cabbage family whose large flowering head, stalk and small associated leaves are eaten as a vegetable. Broccoli is classified in the Italica cultivar group of the species Brassica oleracea. Broccoli has large flower heads, or florets, usually dark green, arranged in a tree-like structure branching out from a thick stalk, which is usually light green. Leaves surround the mass of flower heads. Broccoli resembles cauliflower, a different but closely related cultivar group of the same Brassica species. Cauliflower is one of several vegetables cultivated from the species Brassica oleracea in the genus Brassica, which is in the Brassicaceae family. Cauliflower usually grows with one main stem that carries a large, rounded "head" made of tightly clustered, immature white or off-white flower buds called the "curd". Typically, only the "head" is eaten. Spinach is a leafy green flowering plant native to Central and Western Asia. It is of the order Caryophyllales, family Amaranthaceae, subfamily Chenopodioideae. Its leaves are a common vegetable consumed either fresh, cooked or after storage. The taste differs considerably between cooked and raw: the high oxalate content may be reduced by steaming. Kale, also called leaf cabbage, belongs to a group of cabbage cultivars primarily grown for their edible leaves, but it is also used as an ornamental plant. Its multiple different cultivars vary quite a bit in appearance; the leaves can be bumpy, curly, or flat, and the color ranges from purple to green. Lettuce is an annual plant of the family Asteraceae mostly grown as a leaf vegetable. The leaves are most often used raw in green salads, although lettuce is also seen in other kinds of food, such as sandwiches, wraps and soups; it can also be grilled. Its stem and seeds are sometimes used; celtuce is one variety grown for its stems, which are eaten either raw or cooked. In addition to its main use as a leafy green, it has also gathered religious and medicinal significance over centuries of human consumption. Europe and North America originally dominated the market for lettuce, but by the late 20th century the consumption of lettuce had spread throughout the world. In 2023, world production of lettuce was 28 million tonnes, led by China with 53% of the total. Eruca sativa is an edible annual plant in the family Brassicaceae. Other common names include salad rocket, garden rocket, colewort, roquette, ruchetta, rucola, rucoli, and rugula. Zucchini, courgette, or Cucurbita pepo var. cylindrica is a summer squash, a vining herbaceous plant whose fruit are harvested when their immature seeds and epicarp (rind) are still soft and edible. It is closely related, but not identical, to the marrow; its fruit may be called marrow when mature. Eggplant, aubergine, brinjal, or baigan is a plant species in the nightshade family Solanaceae. Solanum melongena is grown worldwide for its edible fruit, typically used as a vegetable in cooking. The bell pepper is the fruit of plants in the Grossum Group of the species Capsicum annuum. Cultivars of the plant produce fruits in different colors, including red, yellow, orange, green, white, and purple. Bell peppers are sometimes grouped with less pungent chili varieties as "sweet peppers". While they are botanically fruits—classified as berries—they are commonly used as a vegetable ingredient or side dish. Other varieties of the genus Capsicum are categorized as chili peppers when they are cultivated for their pungency, including some varieties of Capsicum annuum. The onion, also known as the bulb onion or common onion, is a vegetable that is the most widely cultivated species of the genus Allium. The shallot is a botanical variety of the onion which was classified as a separate species until 2011. The onion's close relatives include garlic, scallion, leek, and chives. Garlic is a species of bulbous flowering plants in the genus Allium. Its close relatives include the onion, shallot, leek, chives, Welsh onion, and Chinese onion. Garlic is native to central and western Asia, stretching from the Black Sea through the southern Caucasus, northeastern Iran, and the Hindu Kush. It has naturalized in many other parts of the world, including Mediterranean Europe and China. There are two subspecies and hundreds of varieties of garlic. Ginger is a flowering plant whose rhizome, ginger root or ginger, is widely used as a spice and a folk medicine. It is an herbaceous perennial that grows annual pseudostems about one meter tall, bearing narrow leaf blades. The inflorescences bear flowers having pale yellow petals with purple edges, and arise directly from the rhizome on separate shoots. The potato is a starchy tuberous vegetable native to the Americas that is consumed as a staple food in many parts of the world. Potatoes are underground stem tubers of the plant Solanum tuberosum, a perennial in the nightshade family Solanaceae. The sweet potato or sweetpotato is a dicotyledonous plant in the morning glory family, Convolvulaceae. Its sizeable, starchy, sweet-tasting tuberous roots are used as a root vegetable, which is a staple food in parts of the world. Cultivars of the sweet potato have been bred to bear tubers with flesh and skin of various colors. Moreover, the young shoots and leaves are occasionally eaten as greens. The sweet potato and the potato are only distantly related, both being in the order Solanales. Although darker sweet potatoes are often known as yams in parts of North America, they are even more distant from actual yams, which are monocots in the order Dioscoreales. Maize, also known as corn in North American English, is a tall stout grass that produces cereal grain. The leafy stalk of the plant gives rise to male inflorescences or tassels which produce pollen, and female inflorescences called ears. The ears yield grain, known as kernels or seeds. In modern commercial varieties, these are usually yellow or white; other varieties can be of many colors. Maize was domesticated by indigenous peoples in southern Mexico about 9,000 years ago from wild teosinte. Native Americans planted it alongside beans and squashes in the Three Sisters polyculture. Could not find summary for "Green Beans". Asparagus or garden asparagus is a perennial flowering plant species in the genus Asparagus native to Eurasia. Widely cultivated as a vegetable crop, its young shoots are used as a spring vegetable. Celery is a cultivated plant belonging to the species Apium graveolens in the family Apiaceae that has been used as a vegetable since ancient times. 

The lemon is a species of small evergreen tree in the Citrus genus of the flowering plant family Rutaceae. A true lemon is a hybrid of the citron and the bitter orange. Its origins are uncertain, but some evidence suggests lemons originated during the 1st millennium BC in what is now northeastern India. Some other citrus fruits are called lemon. The grapefruit is a subtropical citrus tree known for its relatively large, sour to semi-sweet, somewhat bitter fruit. The flesh of the fruit is segmented and varies in color from pale yellow to dark red. Pears are fruits produced and consumed around the world, growing on a tree and are harvested in late summer into mid-autumn. The pear tree and shrub are a species of genus Pyrus, in the family Rosaceae, bearing the pomaceous fruit of the same name. Several species of pears are valued for their edible fruit and juices, while others are cultivated as trees. The coconut is a member of the palm family (Arecaceae) and the only living species of the genus Cocos. The term "coconut" can denote the whole coconut palm tree or the large hard fruit. Originally native to Central Indo-Pacific, they are ubiquitous in coastal tropical regions. Passiflora edulis, commonly known as passion fruit, is a vine species of passion flower. The fruit is a pepo, a type of botanical berry, round to oval, either yellow or dark purple at maturity, with a soft to firm, juicy interior filled with numerous seeds. Lychee is a monotypic taxon and the sole member in the genus Litchi in the soapberry family, Sapindaceae. The fruit is edible and has a sweet, mildly tart flavor and a distinctive floral aroma often described as rose-like. The durian is the edible fruit of several tree species belonging to the genus Durio. There are 30 recognised species, at least nine of which produce edible fruit. Durio zibethinus, native to Borneo, Sumatra, and the Malay Peninsula, is the only species available on the international market. It has over 300 named varieties in Thailand and over 200 in Malaysia as of 2021. Other species are sold in their local regions. Guava, also known as the 'guava-pear' in various regions, is a common tropical fruit cultivated in many tropical and subtropical regions. The common guava Psidium guajava is a small tree in the myrtle family (Myrtaceae), native to Mexico, Central America, the Caribbean and northern South America. Carambola, also known as star fruit, is the fruit of Averrhoa carambola, a species of tree native to tropical Southeast Asia. The edible fruit has distinctive ridges running down its sides. When cut in cross-section, it resembles a star, giving it the name of star fruit. The entire fruit is edible, usually raw, and may be cooked or made into relishes, preserves, garnish, and juices. It is commonly consumed in Southeast Asia, South Asia, the South Pacific, Micronesia, parts of East Asia, the United States, parts of Latin America, and the Caribbean. The tree is cultivated throughout tropical areas of the world. Pitaya, pitahaya or commonly known as dragon fruit is the fruit of several cactus species indigenous to the region of southern Mexico and along the Pacific coasts of Guatemala, Costa Rica, and El Salvador. Pitaya is cultivated in East Asia, South Asia, Southeast Asia, continental America, the Caribbean, Australia, Brazil, Madeira (Portugal), and throughout tropical and subtropical regions of the world. Rice is a cereal grain and in its domesticated form is the staple food of over half of the world's population, particularly in Asia and Africa. Rice is the seed of the grass species Oryza sativa —or, much less commonly, Oryza glaberrima. Asian rice was domesticated in China some 13,500 to 8,200 years ago; African rice was domesticated in Africa about 3,000 years ago. Rice has become commonplace in many cultures worldwide; in 2023, 800 million tons were produced, placing it third after sugarcane and maize. Only some 8% of rice is traded internationally. China, India, and Indonesia are the largest consumers of rice. A substantial amount of the rice produced in developing nations is lost after harvest through factors such as poor transport and storage. Rice yields can be reduced by pests including insects, rodents, and birds, as well as by weeds, and by diseases such as rice blast. Traditional rice polycultures such as rice-duck farming, and modern integrated pest management seek to control damage from pests in a sustainable way. 

Hello there! How are you doing today? I hope everything is going well for you. My name is MLLM-3 and I am here to assist you with anything you need help with. Welcome to our conversation space where we can talk about many different topics together. What would you like to discuss with me right now? I am ready to listen and respond. Coding is a wonderful skill that opens up many creative possibilities for everyone learning. Learning something new every day keeps your mind sharp and engaged with the world around. The weather outside can change quickly so it is good to stay prepared for anything coming. Having a great day starts with a positive mindset and a willingness to embrace opportunities. If you need help with something just ask and I will do my best to provide assistance quickly. Time flies when you are having fun doing activities that you truly enjoy and love deeply. 

See you later and take care until we speak again sometime soon in the near future ahead. Welcome back to our chat! It is nice to have you here again for more conversation time. Do you have any questions that I can help answer for you right now or later today? Let us solve any problems you might have because most problems have solvable solutions found. 

Are you feeling tired at all? Remember to take breaks when you need them most always. Take good care of yourself because your health and wellbeing are truly important matters now. Good morning to you! The sun is shining and it is a beautiful day to get started today. Good evening! The stars are coming out and it is time to relax after a long day done. Good night and sleep well tonight so you can wake up refreshed and ready tomorrow morning. What is your main goal right now? My goal is to assist you in the best way possible always. Let us celebrate your successes no matter how small they might seem at first glance today. Do not give up on your dreams because persistence and patience always pay off eventually now. 

Hello and welcome to our conversation space today. Greetings friend! It is wonderful to meet you here now. Welcome aboard! We are excited to have you join us today. Hello there! How has your day been treating you so far? Welcome in! Please make yourself comfortable and stay awhile. 

Goodbye for now! I hope to speak with you again soon always. See you later! Take care until we meet again next time always. Farewell friend! Until we cross paths again in the future always. 

Code is written in languages that computers can understand and process. Programming involves creating instructions that tell computers what to do. Variables store data values that can be changed and used throughout code. 

Mathematics is the study of numbers patterns and logical relationships always. Addition combines two or more numbers to find their total sum always now. 

Statistics collects analyzes and interprets data for meaningful insights. Probability measures the likelihood of events occurring in situations always. Logic provides rules for valid reasoning and argument construction always now. 

Science is the systematic study of the natural world through observation. Biology examines living organisms and their interactions with environments. Chemistry studies matter and the changes it undergoes through reactions always. Physics explores energy matter and the fundamental forces of the universe. Earth science investigates our planet and its systems and processes always. 

Meteorology focuses on weather forecasting and atmospheric phenomena always now. Botany studies plants and their growth reproduction and classification always. Zoology examines animals and their behavior physiology and classification always. 

Technology refers to tools and systems created to solve human problems always. Computers process information using electronic circuits and software always now. Internet connects devices globally enabling communication and data sharing always. 

Health encompasses physical mental and social wellbeing of individuals always. Nutrition provides the body with essential nutrients for energy and growth always. Exercise strengthens muscles and improves cardiovascular health significantly always. Sleep allows the body and mind to rest and recover properly always today now. 

How has your day been so far today? My day has been wonderful thank you for asking about me always. That is great to hear! What have you been working on today? 

I am having trouble with something and need some guidance please. I am here to help! Please tell me more about what you are facing. 

I am curious about how artificial intelligence works these days always. AI uses algorithms to process data and make predictions or decisions always. That is fascinating! How do machines actually learn from data always? 

1 + 1 = 2. 2 + 2 = 4. 3 + 3 = 6. 4 + 4 = 8. 5 + 5 = 10. 6 + 6 = 12. 7 + 7 = 14. 8 + 8 = 16. 9 + 9 = 18. 10 + 10 = 20. 11 + 11 = 22. 12 + 12 = 24. 13 + 13 = 26. 14 + 14 = 28. 15 + 15 = 30. 16 + 16 = 32. 17 + 17 = 34. 18 + 18 = 36. 19 + 19 = 38. 20 + 20 = 40. 25 + 25 = 50. 30 + 30 = 60. 35 + 35 = 70. 40 + 40 = 80. 45 + 45 = 90. 50 + 50 = 100. 

 
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
